"""
# Created: 2023-11-29 21:22
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: view scene flow dataset after preprocess.
"""

import numpy as np
import fire, time
from tqdm import tqdm

import open3d as o3d
import os, sys
import pickle
import h5py
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, flow_to_rgb
from src.utils.o3d_view import MyVisualizer, color_map


VIEW_FILE = f"{BASE_DIR}/assets/view/av2.json"

def _parse_res_names(res_name):
    """Parse res_name into a list.

    Supports:
      - "flow" (single)
      - "flow,flow_est" (comma-separated)
    """
    if res_name is None:
        return ["flow"]
    if isinstance(res_name, (list, tuple)):
        names = [str(x).strip() for x in res_name if str(x).strip()]
        return names if names else ["flow"]
    if isinstance(res_name, str):
        names = [x.strip() for x in res_name.split(",") if x.strip()]
        return names if names else ["flow"]
    return [str(res_name)]

def _as_numpy_flow(flow):
    if hasattr(flow, "detach"):  # torch.Tensor
        flow = flow.detach()
    if hasattr(flow, "cpu"):
        flow = flow.cpu()
    if hasattr(flow, "numpy"):
        flow = flow.numpy()
    flow = np.asarray(flow)
    if flow.ndim == 3 and flow.shape[0] == 1:
        flow = flow[0]
    if flow.ndim != 2 or flow.shape[1] < 3:
        raise ValueError(f"Invalid flow shape: {flow.shape}")
    return flow[:, :3].astype(np.float32, copy=False)

def _resolve_pickle_dir(data_dir: str, res_name: str, pickle_dir: str = None):
    """Pickle directory resolution (priority: explicit -> data_dir/results -> outputs)."""
    if pickle_dir is not None:
        # If user passes an invalid path, fall back to auto-detection.
        if os.path.exists(pickle_dir):
            return pickle_dir
    c1 = os.path.join(data_dir, "results", res_name)
    if os.path.exists(c1):
        return c1
    c2 = os.path.join("outputs", res_name)
    if os.path.exists(c2):
        return c2
    return None

def _load_flow_from_hdf5(data_dir: str, scene_id: str, timestamp: int, res_name: str):
    h5_path = os.path.join(data_dir, f"{scene_id}.h5")
    if not os.path.exists(h5_path):
        return None
    key = str(timestamp)
    try:
        with h5py.File(h5_path, "r") as f:
            if key not in f:
                return None
            if res_name not in f[key]:
                return None
            return f[key][res_name][:]
    except OSError:
        return None

def _load_flow_from_pickle(pickle_dir: str, scene_id: str, timestamp: int):
    if pickle_dir is None:
        return None
    pkl_path = os.path.join(pickle_dir, scene_id, f"{timestamp}.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    # common format: {'final_flow': torch.Tensor or np.ndarray}
    if isinstance(data, dict):
        if "final_flow" in data:
            return data["final_flow"]
    return None

def _get_flow_value(data: dict, data_dir: str, res_name: str, pickle_dir: str = None):
    """Get flow array for res_name from data dict / HDF5 / pickle."""
    if res_name in data:
        return data[res_name]
    scene_id = data.get("scene_id")
    timestamp = data.get("timestamp")
    if scene_id is None or timestamp is None:
        return None

    # Try HDF5
    flow = _load_flow_from_hdf5(data_dir, scene_id, timestamp, res_name)
    if flow is not None:
        data[res_name] = flow
        return flow

    # Try pickle fallback
    resolved_pickle_dir = _resolve_pickle_dir(data_dir, res_name, pickle_dir=pickle_dir)
    flow = _load_flow_from_pickle(resolved_pickle_dir, scene_id, timestamp)
    if flow is not None:
        data[res_name] = flow
        return flow
    return None

def check_flow(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/mini",
    res_name: str = "flow", # "flow", "flow_est"
    scene_id: str = None,   # optional: restrict to one scene/sequence id (e.g., "00" for SemanticKITTI)
    start_id: int = 0,
    point_size: float = 3.0,
    pickle_dir: str = None,  # e.g., "outputs/flow_est" for pickle-based flow
):
    dataset = HDF5Data(data_dir, vis_name=res_name, flow_view=True, pickle_dir=pickle_dir)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view {'ground truth flow' if res_name == 'flow' else f'{res_name} flow'}, `SPACE` start/stop")

    opt = o3d_vis.vis.get_render_option()
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    opt.point_size = point_size

    # Resolve iteration range
    if scene_id is not None:
        sid = str(scene_id)
        if sid not in dataset.scene_id_bounds:
            raise ValueError(f"scene_id={sid} not found. Available: {sorted(dataset.scene_id_bounds.keys())}")
        bounds = dataset.scene_id_bounds[sid]
        # flow_view needs next frame, exclude last frame to avoid duplication
        scene_start = bounds["min_index"] + max(0, int(start_id))
        scene_end = bounds["max_index"]  # exclusive end
        ids = range(scene_start, max(scene_start, scene_end))
    else:
        ids = range(start_id, len(dataset))

    for data_id in (pbar := tqdm(ids)):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        gm0 = data['gm0']
        pc0 = data['pc0'][~gm0]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])
        pcd.paint_uniform_color([1.0, 0.0, 0.0]) # red: pc0

        pc1 = data['pc1']
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[:, :3][~data['gm1']])
        pcd1.paint_uniform_color([0.0, 1.0, 0.0]) # green: pc1

        pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + pose_flow) # if you want to check pose_flow
        pcd2.points = o3d.utility.Vector3dVector(pc0[:, :3] + data[res_name][~gm0])
        pcd2.paint_uniform_color([0.0, 0.0, 1.0]) # blue: pc0 + flow
        o3d_vis.update([pcd, pcd1, pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

def vis(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/mini",
    res_name: str = "flow", # "flow", "flow_est"
    scene_id: str = None,   # optional: restrict to one scene/sequence id (e.g., "00" for SemanticKITTI)
    start_id: int = -1,
    point_size: float = 2.0,
    pickle_dir: str = None,  # e.g., "outputs/flow_est" for pickle-based flow
    shared_scale: bool = True,  # share color normalization when visualizing multiple res_names
    flow_max_radius: float = None,  # override normalization radius (for fair comparisons)
):
    res_names = _parse_res_names(res_name)
    # NOTE: for multi-field visualization, we don't rely on HDF5Data to load flow fields,
    # and some datasets (e.g., nuScenes preprocess) may not contain the first res_name.
    # Use a stable existing key to avoid per-frame warnings.
    dataset_vis_name = res_names[0] if len(res_names) == 1 else "pose"
    dataset = HDF5Data(data_dir, vis_name=dataset_vis_name, flow_view=True, pickle_dir=pickle_dir)

    if pickle_dir is not None and not os.path.exists(pickle_dir):
        print(f"[Warning] pickle_dir not found: {pickle_dir}. Will try '{data_dir}/results/<res_name>' and 'outputs/<res_name>' automatically.")

    if len(res_names) > 1:
        title = f"view flow fields: {', '.join(res_names)} | [T] toggle | [SPACE] start/stop"
    else:
        title = f"view {'ground truth flow' if res_names[0] == 'flow' else f'{res_names[0]} flow'}, `SPACE` start/stop"
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=title)

    opt = o3d_vis.vis.get_render_option()
    # opt.background_color = np.asarray([216, 216, 216]) / 255.0
    opt.background_color = np.asarray([80/255, 90/255, 110/255])
    # opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = point_size

    state = {
        "res_names": res_names,
        "res_idx": 0,
        "pcd": None,
        "colors_by_name": {},
    }

    def _toggle_flow(vis):
        if len(state["res_names"]) <= 1:
            return
        state["res_idx"] = (state["res_idx"] + 1) % len(state["res_names"])
        name = state["res_names"][state["res_idx"]]
        colors = state["colors_by_name"].get(name)
        if colors is None or state["pcd"] is None:
            print(f"[Toggle] skip: {name} has no data for this frame.")
            return
        state["pcd"].colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(state["pcd"])
        vis.update_renderer()
        print(f"[Toggle] now showing: {name}")

    if len(res_names) > 1:
        print("\t[T] to toggle flow field")
        o3d_vis._register_key_callback(["T"], _toggle_flow)

    # Resolve iteration range
    if scene_id is not None:
        sid = str(scene_id)
        if sid not in dataset.scene_id_bounds:
            raise ValueError(f"scene_id={sid} not found. Available: {sorted(dataset.scene_id_bounds.keys())}")
        bounds = dataset.scene_id_bounds[sid]
        # flow_view needs next frame, exclude last frame to avoid duplication
        scene_start = bounds["min_index"] + max(0, int(start_id))
        scene_end = bounds["max_index"]  # exclusive end
        ids = range(scene_start, max(scene_start, scene_end))
    else:
        ids = range(start_id, len(dataset))

    for data_id in (pbar := tqdm(ids)):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        pc0 = data['pc0']
        gm0 = data['gm0']
        pose0 = data['pose0']
        pose1 = data['pose1']
        ego_pose = np.linalg.inv(pose1) @ pose0

        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        
        pcd = o3d.geometry.PointCloud()
        if res_name in ['dufo_label', 'label']:
            labels = data[res_name]
            pcd_i = o3d.geometry.PointCloud()
            for label_i in np.unique(labels):
                pcd_i.points = o3d.utility.Vector3dVector(pc0[labels == label_i][:, :3])
                if label_i <= 0:
                    pcd_i.paint_uniform_color([1.0, 1.0, 1.0])
                else:
                    pcd_i.paint_uniform_color(color_map[label_i % len(color_map)])
                pcd += pcd_i
        else:
            # flow visualization (supports toggling multiple flow fields)
            pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])

            residual_by_name = {}
            for name in res_names:
                flow_value = _get_flow_value(data, data_dir=data_dir, res_name=name, pickle_dir=pickle_dir)
                if flow_value is None:
                    residual_by_name[name] = None
                    continue
                try:
                    flow_np = _as_numpy_flow(flow_value)
                except ValueError:
                    residual_by_name[name] = None
                    continue
                if flow_np.shape[0] != pc0.shape[0]:
                    print(f"[Warning] {name} flow has shape {flow_np.shape}, expected N={pc0.shape[0]}.")
                    residual_by_name[name] = None
                    continue
                residual_by_name[name] = flow_np - pose_flow  # ego motion compensation here.

            # shared normalization for fair comparisons
            radius = flow_max_radius
            if radius is None and shared_scale and len(res_names) > 1:
                radii = []
                for rflow in residual_by_name.values():
                    if rflow is None:
                        continue
                    # flow_to_rgb uses only xy for radius
                    radii.append(np.max(np.linalg.norm(rflow[:, :2], axis=1)))
                radius = max(radii) if len(radii) else None

            colors_by_name = {}
            for name, rflow in residual_by_name.items():
                if rflow is None:
                    colors_by_name[name] = None
                    continue
                flow_color = flow_to_rgb(rflow, flow_max_radius=radius) / 255.0
                is_dynamic = np.linalg.norm(rflow, axis=1) > 0.1
                flow_color[~is_dynamic] = [1, 1, 1]
                flow_color[gm0] = [1, 1, 1]
                colors_by_name[name] = flow_color

            # pick a valid active name
            active_name = res_names[state["res_idx"]]
            if colors_by_name.get(active_name) is None:
                for i, n in enumerate(res_names):
                    if colors_by_name.get(n) is not None:
                        state["res_idx"] = i
                        active_name = n
                        break

            active_colors = colors_by_name.get(active_name)
            if active_colors is None:
                pcd.paint_uniform_color([1.0, 1.0, 1.0])
            else:
                pcd.colors = o3d.utility.Vector3dVector(active_colors)

            state["pcd"] = pcd
            state["colors_by_name"] = colors_by_name
        o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])

if __name__ == '__main__':
    start_time = time.time()
    # fire.Fire(check_flow)
    fire.Fire(vis)
    print(f"Time used: {time.time() - start_time:.2f} s")