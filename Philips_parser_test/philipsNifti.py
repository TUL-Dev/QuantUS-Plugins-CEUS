"""
ULTRA-FAST NIfTI Converter for Philips Volumes
Optimized for speed with batch I/O and parallel processing
"""

import os
from pathlib import Path
import pickle
import numpy as np
import nibabel as nib
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def load_single_frame(frame_info):
    """Load a single frame - used for parallel processing"""
    frame_path, frame_num = frame_info
    try:
        with open(frame_path, 'rb') as f:
            frame_data = pickle.load(f)
        return frame_num, np.clip(frame_data, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error loading frame {frame_num}: {e}")
        return frame_num, None


def format_time_series_fast(dest_folder: str, data_name: str, bmode: bool, 
                            n_workers: int = 4) -> str:
    """
    Ultra-fast conversion with parallel frame loading
    
    Args:
        dest_folder: Folder containing pickle files
        data_name: Base name for output file
        bmode: True for B-mode, False for CEUS
        n_workers: Number of parallel workers for loading (default: 4)
    
    Returns:
        Path to saved NIfTI file
    """
    if bmode:
        print("Converting B-Mode volumes to NIfTI...")
        frame_prefix = "bmode_frame"
        output_suffix = "_BMODE.nii.gz"
    else:
        print("Converting CEUS volumes to NIfTI...")
        frame_prefix = "ceus_frame"
        output_suffix = "_CEUS.nii.gz"
    
    dest_path = Path(dest_folder)
    
    # Load resolution information
    res_file_path = dest_path / "bmode_volume_dims.pkl"
    with open(res_file_path, 'rb') as f:
        org_res = pickle.load(f)
    
    print("Scanning for frames...")
    # Get all frame files
    frame_files = sorted([f for f in dest_path.iterdir() 
                         if f.name.startswith(frame_prefix) and f.name.endswith(".pkl")])
    
    num_frames = len(frame_files)
    print(f"Found {num_frames} frames")
    
    if num_frames == 0:
        raise ValueError(f"No frames found with prefix '{frame_prefix}'")
    
    # Load first frame to get dimensions
    with open(frame_files[0], 'rb') as f:
        first_frame = pickle.load(f)
    
    frame_shape = first_frame.shape
    output_shape = frame_shape + (num_frames,)
    
    print(f"Frame shape: {frame_shape}")
    print(f"Output shape: {output_shape}")
    
    # Estimate memory
    total_size_mb = (np.prod(output_shape)) / (1024**2)
    print(f"Total size: {total_size_mb:.1f} MB")
    
    # Pre-allocate output array
    print("Pre-allocating array...")
    time_series_vols = np.zeros(output_shape, dtype=np.uint8)
    
    # Store first frame (already loaded)
    time_series_vols[..., 0] = np.clip(first_frame, 0, 255).astype(np.uint8)
    
    # Parallel loading of remaining frames
    if n_workers > 1 and num_frames > 10:
        print(f"Loading frames in parallel (using {n_workers} workers)...")
        
        # Prepare frame info for parallel processing (skip first frame)
        frame_info_list = [(frame_files[i], i) for i in range(1, num_frames)]
        
        # Use multiprocessing pool
        with mp.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(load_single_frame, frame_info_list),
                total=len(frame_info_list),
                desc="Loading frames"
            ))
        
        # Store results
        for frame_num, frame_data in results:
            if frame_data is not None:
                time_series_vols[..., frame_num] = frame_data
    
    else:
        # Sequential loading (for small datasets or single worker)
        print("Loading frames sequentially...")
        for i in tqdm(range(1, num_frames), desc="Loading frames"):
            with open(frame_files[i], 'rb') as f:
                frame_data = pickle.load(f)
            time_series_vols[..., i] = np.clip(frame_data, 0, 255).astype(np.uint8)
    
    print("Creating NIfTI image...")
    
    # Create affine matrix
    affine = np.diag([org_res[1], org_res[2], org_res[3], 1])
    
    # Create NIfTI image
    nii_array = nib.Nifti1Image(time_series_vols, affine)
    nii_array.header['pixdim'] = org_res
    nii_array.header['xyzt_units'] = 10  # mm and seconds
    
    # Save
    output_path = dest_path / (data_name.replace('.raw', '') + output_suffix)
    print(f"Saving NIfTI file: {output_path}")
    nib.save(nii_array, str(output_path))
    
    print(f"✓ Saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.1f} MB")
    
    return str(output_path)


def format_time_series_batch(dest_folder: str, data_name: str, bmode: bool,
                             batch_size: int = 50) -> str:
    """
    Batch loading version - loads frames in batches to reduce I/O overhead
    Good balance between speed and memory usage
    
    Args:
        dest_folder: Folder containing pickle files
        data_name: Base name for output file
        bmode: True for B-mode, False for CEUS
        batch_size: Number of frames to load at once (default: 50)
    
    Returns:
        Path to saved NIfTI file
    """
    if bmode:
        print("Converting B-Mode volumes to NIfTI (batch mode)...")
        frame_prefix = "bmode_frame"
        output_suffix = "_BMODE.nii.gz"
    else:
        print("Converting CEUS volumes to NIfTI (batch mode)...")
        frame_prefix = "ceus_frame"
        output_suffix = "_CEUS.nii.gz"
    
    dest_path = Path(dest_folder)
    
    # Load resolution
    with open(dest_path / "bmode_volume_dims.pkl", 'rb') as f:
        org_res = pickle.load(f)
    
    # Get frame files
    frame_files = sorted([f for f in dest_path.iterdir() 
                         if f.name.startswith(frame_prefix) and f.name.endswith(".pkl")])
    
    num_frames = len(frame_files)
    print(f"Found {num_frames} frames")
    
    # Load first frame for dimensions
    with open(frame_files[0], 'rb') as f:
        first_frame = pickle.load(f)
    
    frame_shape = first_frame.shape
    output_shape = frame_shape + (num_frames,)
    
    print(f"Output shape: {output_shape}")
    print(f"Batch size: {batch_size} frames")
    
    # Pre-allocate
    time_series_vols = np.zeros(output_shape, dtype=np.uint8)
    
    # Process in batches
    num_batches = int(np.ceil(num_frames / batch_size))
    
    print(f"Processing {num_batches} batches...")
    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_frames)
        
        # Load batch
        batch_frames = []
        for i in range(start_idx, end_idx):
            with open(frame_files[i], 'rb') as f:
                frame_data = pickle.load(f)
            batch_frames.append(np.clip(frame_data, 0, 255).astype(np.uint8))
        
        # Store batch
        batch_array = np.stack(batch_frames, axis=-1)
        time_series_vols[..., start_idx:end_idx] = batch_array
    
    print("Creating NIfTI image...")
    
    # Create and save NIfTI
    affine = np.diag([org_res[1], org_res[2], org_res[3], 1])
    nii_array = nib.Nifti1Image(time_series_vols, affine)
    nii_array.header['pixdim'] = org_res
    nii_array.header['xyzt_units'] = 10
    
    output_path = dest_path / (data_name.replace('.raw', '') + output_suffix)
    print(f"Saving: {output_path}")
    nib.save(nii_array, str(output_path))
    
    print(f"✓ Complete: {output_path}")
    return str(output_path)


def makeNifti(dest_folder, data_name, mode='parallel', n_workers=4, batch_size=50):
    """
    Fast NIfTI creation with multiple optimization modes
    
    Args:
        dest_folder: Folder containing pickle files
        data_name: Base name for output
        mode: 'parallel' (fastest, uses multiple cores) or 'batch' (balanced)
        n_workers: Number of workers for parallel mode
        batch_size: Batch size for batch mode
    
    Returns:
        Tuple of (ceus_path, bmode_path)
    """
    import time
    
    print("=" * 70)
    print("FAST NIFTI CONVERTER")
    print("=" * 70)
    print(f"Mode: {mode.upper()}")
    print(f"Data folder: {dest_folder}")
    print(f"Output name: {data_name}")
    print("=" * 70)
    
    if mode == 'parallel':
        print(f"\nUsing PARALLEL mode with {n_workers} workers")
        
        start_time = time.time()
        ceus_path = format_time_series_fast(dest_folder, data_name, 
                                           bmode=False, n_workers=n_workers)
        ceus_time = time.time() - start_time
        print(f"CEUS conversion took: {ceus_time:.1f} seconds")
        
        start_time = time.time()
        bmode_path = format_time_series_fast(dest_folder, data_name, 
                                            bmode=True, n_workers=n_workers)
        bmode_time = time.time() - start_time
        print(f"B-mode conversion took: {bmode_time:.1f} seconds")
        
    elif mode == 'batch':
        print(f"\nUsing BATCH mode with batch_size={batch_size}")
        
        start_time = time.time()
        ceus_path = format_time_series_batch(dest_folder, data_name, 
                                            bmode=False, batch_size=batch_size)
        ceus_time = time.time() - start_time
        print(f"CEUS conversion took: {ceus_time:.1f} seconds")
        
        start_time = time.time()
        bmode_path = format_time_series_batch(dest_folder, data_name, 
                                             bmode=True, batch_size=batch_size)
        bmode_time = time.time() - start_time
        print(f"B-mode conversion took: {bmode_time:.1f} seconds")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'parallel' or 'batch'")
    
    total_time = ceus_time + bmode_time
    print("\n" + "=" * 70)
    print(f"✓ COMPLETE - Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("=" * 70)
    print(f"CEUS: {ceus_path}")
    print(f"B-mode: {bmode_path}")
    
    return ceus_path, bmode_path

if __name__ == "__main__":
    destFolder = Path('/Volumes/T5 EVO/UCSD_3DCEUS/SIP/2025.08.26_P03V03/TestInterpolation/UCSD-P03-V03-CE1_10.37.08')
    dataName = 'UCSD-P03-V03-CE1_10.37.08_mf_sip_capture_50_2_1_0.raw'
    makeNifti(destFolder, dataName)
    