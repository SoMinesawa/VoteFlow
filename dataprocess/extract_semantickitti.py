"""
# Created: 2024-12-28
# Description: Preprocess SemanticKITTI data for VoteFlow, save as HDF5 format
# 
# SemanticKITTI data structure:
#   sequences/XX/velodyne/*.bin  - point cloud (N, 4) float32 (x, y, z, intensity)
#   sequences/XX/poses.txt       - poses (4x4 matrix per line, 12 values)
#   sequences/XX/labels/*.label  - semantic labels (optional)
# 
# Patchwork++ ground mask structure:
#   sequences/XX/predictions/*.label - ground mask (N,) uint32 (1=ground, 0=non-ground)
"""

import os
import numpy as np
import h5py
import pickle
import fire
import time
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_velodyne_scan(scan_path: str) -> np.ndarray:
    """Load a velodyne scan from binary file.
    
    Args:
        scan_path: Path to .bin file
        
    Returns:
        Point cloud as (N, 4) array [x, y, z, intensity]
    """
    scan = np.fromfile(scan_path, dtype=np.float32)
    return scan.reshape((-1, 4))


def load_poses(pose_file: str) -> np.ndarray:
    """Load poses from KITTI format poses.txt file.
    
    Args:
        pose_file: Path to poses.txt
        
    Returns:
        Array of shape (num_frames, 4, 4) containing SE3 poses
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            if len(values) == 12:
                # KITTI format: 3x4 matrix in row-major order
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :] = np.array(values).reshape(3, 4)
                poses.append(pose)
    return np.array(poses)


def load_patchwork_ground_mask(label_path: str) -> np.ndarray:
    """Load Patchwork++ ground mask from .label file.
    
    Args:
        label_path: Path to .label file
        
    Returns:
        Boolean array where True = ground point
    """
    labels = np.fromfile(label_path, dtype=np.uint32)
    # Patchwork++ outputs: 1 = ground, 0 = non-ground
    return labels == 1


def process_sequence(args):
    """Process a single sequence.
    
    Args:
        args: Tuple of (seq_id, data_dir, patchwork_dir, output_dir)
        
    Raises:
        FileNotFoundError: If Patchwork++ ground mask is not found
    """
    seq_id, data_dir, patchwork_dir, output_dir = args
    
    seq_path = Path(data_dir) / "sequences" / seq_id
    velodyne_dir = seq_path / "velodyne"
    poses_file = seq_path / "poses.txt"
    
    if not velodyne_dir.exists():
        raise FileNotFoundError(f"Sequence {seq_id}: velodyne directory not found at {velodyne_dir}")
    
    if not poses_file.exists():
        raise FileNotFoundError(f"Sequence {seq_id}: poses.txt not found at {poses_file}")
    
    # Check Patchwork++ directory exists
    if not patchwork_dir:
        raise ValueError("patchwork_dir is required. Please provide Patchwork++ ground mask directory.")
    
    patchwork_seq_dir = Path(patchwork_dir) / "sequences" / seq_id / "predictions"
    if not patchwork_seq_dir.exists():
        raise FileNotFoundError(
            f"Sequence {seq_id}: Patchwork++ predictions not found at {patchwork_seq_dir}\n"
            "Please ensure Patchwork++ ground masks are available for all sequences."
        )
    
    # Load poses
    poses = load_poses(str(poses_file))
    
    # Get list of scans
    scan_files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    
    if len(scan_files) == 0:
        raise FileNotFoundError(f"Sequence {seq_id}: no .bin scan files found in {velodyne_dir}")
    
    # Create output HDF5 file
    output_file = Path(output_dir) / f"{seq_id}.h5"
    data_index = []
    
    with h5py.File(output_file, 'w') as f:
        for idx, scan_file in enumerate(tqdm(scan_files, desc=f"Seq {seq_id}", leave=False)):
            frame_id = scan_file.replace('.bin', '')  # e.g., "000000"
            
            # Load point cloud
            scan_path = velodyne_dir / scan_file
            pc = load_velodyne_scan(str(scan_path))[:, :3]  # Only xyz
            
            if pc.shape[0] < 256:
                print(f"Sequence {seq_id}, frame {frame_id}: less than 256 points, skipping")
                continue
            
            # Get pose (use identity if index out of range)
            if idx < len(poses):
                pose = poses[idx]
            else:
                print(f"Sequence {seq_id}, frame {frame_id}: pose index out of range, using identity")
                pose = np.eye(4, dtype=np.float32)
            
            # Load Patchwork++ ground mask (required)
            label_file = patchwork_seq_dir / f"{frame_id}.label"
            if not label_file.exists():
                raise FileNotFoundError(
                    f"Sequence {seq_id}, frame {frame_id}: Patchwork++ label not found at {label_file}"
                )
            ground_mask = load_patchwork_ground_mask(str(label_file))
            
            # Ensure ground_mask matches point cloud size
            if len(ground_mask) != len(pc):
                raise ValueError(
                    f"Sequence {seq_id}, frame {frame_id}: ground mask size ({len(ground_mask)}) "
                    f"does not match point cloud size ({len(pc)})"
                )
            
            # Create group for this frame
            group = f.create_group(frame_id)
            group.create_dataset('lidar', data=pc.astype(np.float32))
            group.create_dataset('ground_mask', data=ground_mask.astype(bool))
            group.create_dataset('pose', data=pose.astype(np.float32))
            
            data_index.append([seq_id, frame_id])
    
    print(f"Sequence {seq_id}: processed {len(data_index)} frames")
    return data_index


def create_reading_index(output_dir: Path, data_index: list):
    """Create index_total.pkl for the dataset.
    
    Args:
        output_dir: Output directory
        data_index: List of [scene_id, timestamp] pairs
    """
    # Sort by scene_id and then by timestamp
    data_index.sort(key=lambda x: (x[0], int(x[1])))
    
    with open(output_dir / 'index_total.pkl', 'wb') as f:
        pickle.dump(data_index, f)
    print(f"Created index_total.pkl with {len(data_index)} entries")


def main(
    kitti_dir: str = "data/dataset/semantickitti/dataset",
    patchwork_dir: str = "data/users/minesawa/semantickitti/patchwork-plusplus",
    output_dir: str = "data/dataset/semantickitti/preprocess",
    sequences: str = None,  # Comma-separated list, e.g., "00,01,02" or None for all
    nproc: int = 1,
):
    """Preprocess SemanticKITTI dataset for VoteFlow.
    
    Args:
        kitti_dir: Path to SemanticKITTI dataset (containing sequences/)
        patchwork_dir: Path to Patchwork++ results (containing sequences/XX/predictions/)
        output_dir: Output directory for preprocessed HDF5 files
        sequences: Comma-separated sequence IDs to process (e.g., "00,01,02"), or None for all
        nproc: Number of parallel processes (default 1, set higher for faster processing)
    """
    kitti_path = Path(kitti_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which sequences to process
    if sequences:
        seq_ids = [s.strip() for s in sequences.split(',')]
    else:
        # Find all sequences
        seq_dir = kitti_path / "sequences"
        if not seq_dir.exists():
            print(f"Error: {seq_dir} does not exist")
            return
        seq_ids = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(seq_dir / d)])
    
    print(f"Processing sequences: {seq_ids}")
    print(f"Input: {kitti_dir}")
    print(f"Patchwork++: {patchwork_dir}")
    print(f"Output: {output_dir}")
    
    # Prepare arguments for multiprocessing
    args_list = [(seq_id, kitti_dir, patchwork_dir, output_dir) for seq_id in seq_ids]
    
    all_data_index = []
    
    if nproc <= 1:
        # Single process
        for args in args_list:
            data_index = process_sequence(args)
            all_data_index.extend(data_index)
    else:
        # Multiprocessing
        with Pool(processes=min(nproc, len(seq_ids))) as pool:
            results = list(tqdm(pool.imap(process_sequence, args_list), 
                               total=len(args_list), desc="Processing sequences"))
            for data_index in results:
                all_data_index.extend(data_index)
    
    # Create index file
    create_reading_index(output_path, all_data_index)
    
    print(f"\nPreprocessing complete!")
    print(f"Output files saved to: {output_dir}")
    print(f"Total frames: {len(all_data_index)}")


if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")

