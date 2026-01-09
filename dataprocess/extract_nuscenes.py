"""
# Created: 2026-01-08
# Description: Preprocess nuScenes data for VoteFlow, save as HDF5 format.
#
# Assumptions:
# - VoteFlow is trained on Waymo (x: forward, y: left, z: up, right-hand).
# - nuScenes LiDAR_TOP shares the same axis convention (x front, y left, z up).
#   Therefore、座標系変換は不要で world_T_lidar をそのまま保存する。
# - 入力フレーム間隔: nuScenes は LIDAR_TOP 20Hz。1フレーム飛ばし（間隔2）で
#   10Hz相当のペアを作るため、index_total には「スタートフレームのみ」を記録し、
#   スタートから +pair_stride フレームを次フレームとみなす。
#   （ペアの実解釈はデータローダ側で pair_stride=2 を指定して行う想定）
#
# 依存:
#   pip install nuscenes-devkit pyquaternion tqdm h5py numpy fire
#
# 出力構造:
#   output_dir/
#     ├── {scene_name}.h5
#     └── index_total.pkl  # [[scene_id, frame_token], ...]  ※pair_strideと併用
#
# Ground mask (Patchwork++):
#   data/users/minesawa/nuscenes/patchwork-plusplus/{samples|sweeps}/LIDAR_TOP/*.label
#   1 = ground, 0 = non-ground
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
from multiprocessing import Pool

import fire
import h5py
import numpy as np
from tqdm import tqdm

# nuScenes devkit
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion
except ImportError as e:
    raise ImportError(
        "nuscenes-devkit と pyquaternion が必要です。pip install nuscenes-devkit pyquaternion"
    ) from e


def quat_trans_to_mat(q: Quaternion, t: np.ndarray) -> np.ndarray:
    """Quaternion + translation -> 4x4 matrix."""
    rot = q.rotation_matrix
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = t
    return mat


def load_lidar(path: Path) -> np.ndarray:
    """nuScenes .bin -> (N,3) float32 (xyz only)."""
    pc = LidarPointCloud.from_file(str(path))
    return pc.points[:3, :].T.astype(np.float32)


def load_ground_mask(label_path: Path) -> np.ndarray:
    """Patchwork++ mask -> bool array (True = ground)."""
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels == 1


def resolve_patchwork_path(patchwork_root: Path, rel_path: str) -> Path:
    """nuScenesファイルの相対パスを.label拡張子に置換してPatchwork++出力を指す。
    
    nuScenes LIDAR_TOP の実データ拡張子は .pcd.bin なので、それを .label に変換する。
    """
    if rel_path.endswith(".pcd.bin"):
        rel_label = rel_path[:-len(".pcd.bin")] + ".label"
    elif rel_path.endswith(".bin"):
        rel_label = rel_path[:-len(".bin")] + ".label"
    else:
        rel_label = rel_path + ".label"
    return patchwork_root / rel_label


def collect_lidar_tokens(nusc: NuScenes, scene_token: str) -> List[str]:
    """シーン内の LIDAR_TOP sample_data を時間順に列挙（20Hz, key+non-key含む）。"""
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    tokens = []
    while sd_token:
        tokens.append(sd_token)
        sd = nusc.get("sample_data", sd_token)
        sd_token = sd["next"]
    return tokens


def build_world_T_lidar(nusc: NuScenes, sd_token: str) -> np.ndarray:
    """world_T_lidar = world_T_ego @ ego_T_lidar."""
    sd = nusc.get("sample_data", sd_token)
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    world_T_ego = quat_trans_to_mat(Quaternion(ego["rotation"]), np.array(ego["translation"], dtype=np.float32))
    ego_T_lidar = quat_trans_to_mat(Quaternion(cal["rotation"]), np.array(cal["translation"], dtype=np.float32))
    return (world_T_ego @ ego_T_lidar).astype(np.float32)


def process_scene(
    nusc: NuScenes,
    scene_name: str,
    scene_token: str,
    data_root: Path,
    patchwork_root: Path,
    output_dir: Path,
    pair_stride: int = 2,
    skip_missing: bool = True,
    min_points: int = 256,
) -> Tuple[List[List[str]], Dict[str, int]]:
    """1シーンをHDF5化し index用リストを返す。"""
    tokens = collect_lidar_tokens(nusc, scene_token)

    output_path = output_dir / f"{scene_name}.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "processed": 0,
        "skipped_missing": 0,
        "skipped_too_few_points": 0,
        "skipped_size_mismatch": 0,
        "errors": 0,
    }

    saved_tokens = set()

    with h5py.File(output_path, "w") as h5f:
        for sd_token in tqdm(tokens, desc=f"Scene {scene_name}", leave=False):
            sd = nusc.get("sample_data", sd_token)
            rel_path = sd["filename"]
            lidar_path = data_root / rel_path
            mask_path = resolve_patchwork_path(patchwork_root, rel_path)

            try:
                if not lidar_path.exists():
                    raise FileNotFoundError(f"LiDAR file missing: {lidar_path}")
                pc = load_lidar(lidar_path)
                if pc.shape[0] < min_points:
                    stats["skipped_too_few_points"] += 1
                    continue

                if not mask_path.exists():
                    if skip_missing:
                        stats["skipped_missing"] += 1
                        continue
                    raise FileNotFoundError(f"Patchwork++ label missing: {mask_path}")

                gm = load_ground_mask(mask_path)
                if len(gm) != pc.shape[0]:
                    if skip_missing:
                        stats["skipped_size_mismatch"] += 1
                        continue
                    raise ValueError(f"Ground mask size mismatch: pc={pc.shape[0]}, mask={len(gm)} at {mask_path}")

                pose = build_world_T_lidar(nusc, sd_token)

                grp = h5f.create_group(sd_token)
                grp.create_dataset("lidar", data=pc, compression="gzip")
                grp.create_dataset("ground_mask", data=gm.astype(bool), compression="gzip")
                grp.create_dataset("pose", data=pose, compression="gzip")

                saved_tokens.add(sd_token)
                stats["processed"] += 1
            except Exception:
                stats["errors"] += 1
                if not skip_missing:
                    raise
                continue

    # index_total は「スタートフレーム」を列挙。スタート + pair_stride が存在し、両方保存できたもののみ。
    data_index = []
    for i in range(len(tokens) - pair_stride):
        t0 = tokens[i]
        t1 = tokens[i + pair_stride]
        if t0 in saved_tokens and t1 in saved_tokens:
            data_index.append([scene_name, t0])

    return data_index, stats


# multiprocessing helpers
_NUSC = None


def _init_worker(nusc_kwargs: dict):
    """Initializer to create a shared NuScenes instance per worker."""
    global _NUSC
    _NUSC = NuScenes(**nusc_kwargs)


def _process_scene_mp(args):
    (
        scene_name,
        scene_token,
        data_root,
        patchwork_root,
        output_root,
        pair_stride,
        skip_missing,
        min_points,
    ) = args
    return process_scene(
        _NUSC,
        scene_name,
        scene_token,
        data_root,
        patchwork_root,
        output_root,
        pair_stride=pair_stride,
        skip_missing=skip_missing,
        min_points=min_points,
    )


def main(
    data_dir: str = "data/dataset/nuscenes",
    version: str = "v1.0-trainval",
    split: str = "trainval",  # train / val / trainval / mini
    patchwork_dir: str = "data/users/minesawa/nuscenes/patchwork-plusplus",
    output_dir: str = "data/dataset/nuscenes/preprocess",
    pair_stride: int = 2,
    skip_missing: bool = True,
    min_points: int = 256,
    nproc: int = 1,
):
    """
    nuScenes を VoteFlow 用に前処理して HDF5 と index_total.pkl を生成する。

    Args:
        data_dir: nuScenes データセットルート
        version: nuScenes バージョン（例 v1.0-trainval, v1.0-mini）
        split: train / val / trainval / mini
        patchwork_dir: Patchwork++ 出力ルート
        output_dir: 保存先
        pair_stride: 1フレーム飛ばしなら2（20Hz->10Hz）
        skip_missing: 欠損/不整合をスキップして続行
        min_points: フレームを採用する最低点数
        nproc: 並列プロセス数（1なら逐次）
    """
    data_root = Path(data_dir)
    patchwork_root = Path(patchwork_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"nuScenes root: {data_root}")
    print(f"Patchwork++ root: {patchwork_root}")
    print(f"Output: {output_root}")
    print(f"pair_stride: {pair_stride} (1frame skip => 10Hz)")
    print(f"split: {split}, version: {version}")
    print(f"skip_missing: {skip_missing}, min_points: {min_points}")

    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)

    if split == "train":
        scene_names = create_splits_scenes()["train"]
    elif split == "val":
        scene_names = create_splits_scenes()["val"]
    elif split == "mini":
        scene_names = create_splits_scenes()["mini_train"] + create_splits_scenes()["mini_val"]
    else:
        scene_names = create_splits_scenes()["train"] + create_splits_scenes()["val"]

    # name -> token
    name_to_token = {scene["name"]: scene["token"] for scene in nusc.scene}

    selected = [(name, name_to_token[name]) for name in scene_names if name in name_to_token]

    all_index = []
    total_stats = {
        "processed": 0,
        "skipped_missing": 0,
        "skipped_too_few_points": 0,
        "skipped_size_mismatch": 0,
        "errors": 0,
    }

    if nproc <= 1:
        for scene_name, scene_token in tqdm(selected, desc="Scenes"):
            data_index, stats = process_scene(
                nusc,
                scene_name,
                scene_token,
                data_root,
                patchwork_root,
                output_root,
                pair_stride=pair_stride,
                skip_missing=skip_missing,
                min_points=min_points,
            )
            all_index.extend(data_index)
            for k, v in stats.items():
                total_stats[k] += v
    else:
        args_list = [
            (
                scene_name,
                scene_token,
                data_root,
                patchwork_root,
                output_root,
                pair_stride,
                skip_missing,
                min_points,
            )
            for scene_name, scene_token in selected
        ]
        with Pool(
            processes=min(nproc, len(selected)),
            initializer=_init_worker,
            initargs=({"version": version, "dataroot": str(data_root), "verbose": False},),
        ) as pool:
            for data_index, stats in tqdm(
                pool.imap_unordered(_process_scene_mp, args_list),
                total=len(args_list),
                desc="Scenes",
            ):
                all_index.extend(data_index)
                for k, v in stats.items():
                    total_stats[k] += v

    # index_total.pkl を保存（シーン名とスタートフレームtoken）
    with open(output_root / "index_total.pkl", "wb") as f:
        pickle.dump(all_index, f)

    print("\n=== Summary ===")
    print(f"Scenes processed: {len(selected)}")
    print(f"Frames saved: {total_stats['processed']}")
    print(f"Skipped (missing Patchwork++): {total_stats['skipped_missing']}")
    print(f"Skipped (too few points < {min_points}): {total_stats['skipped_too_few_points']}")
    print(f"Skipped (mask size mismatch): {total_stats['skipped_size_mismatch']}")
    print(f"Errors: {total_stats['errors']}")
    print(f"index_total entries (pair_stride={pair_stride}): {len(all_index)}")
    print("完了")


if __name__ == "__main__":
    fire.Fire(main)

