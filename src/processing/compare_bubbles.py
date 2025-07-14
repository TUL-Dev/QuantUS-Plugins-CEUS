from typing import Tuple, Dict, List
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes


# Need scan path and mask path for each analysis run

def organize_bolus_data(bolus_data_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Organize bolus data from a given path into a dictionary.
    
    Args:
        bolus_data_path (str): The path to the bolus data file.
        
    Returns:
        Dict[str, str]: A dictionary with keys as 'bolus_data' and 'bolus_path'.
    """
    bolus_data_path = Path(bolus_data_path)
    scan_paths = {}; mask_paths = {}
    for mouse_folder in bolus_data_path.iterdir():
        if mouse_folder.is_dir() and mouse_folder.name.startswith('m'):
            scan_paths[mouse_folder.name] = []
            mask_paths[mouse_folder.name] = []
            for mask_file in mouse_folder.glob('*Full4DMasked.nii'):
                scan_file = mask_file.parent / mask_file.name.replace('Full4DMasked.nii', 'Full4D.nii')
                scan_paths[mouse_folder.name].append(str(scan_file))
                mask_paths[mouse_folder.name].append(str(mask_file))

    return scan_paths, mask_paths

def organize_molecular_data(molecular_data_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Organize molecular data from a given path into a dictionary.
    
    Args:
        molecular_data_path (str): The path to the molecular data file.
        
    Returns:
        Dict[str, str]: A dictionary with keys as 'molecular_data' and 'molecular_path'.
    """
    molecular_data_path = Path(molecular_data_path)
    scan_paths = {}; mask_paths = {}
    for mouse_folder in molecular_data_path.iterdir():
        if mouse_folder.is_dir() and mouse_folder.name.startswith('m'):
            scan_paths[mouse_folder.name] = []
            mask_paths[mouse_folder.name] = [] 
            for scan_file in mouse_folder.glob('*.nii.gz'):
                mask_file = scan_file.parent / 'nifti_segmentation_QUANTUS' / f"{scan_file.name[:-7].replace(' ', '_')}_segmentation.nii.gz"
                if mask_file.exists():
                    scan_paths[mouse_folder.name].append(str(scan_file))
                    mask_paths[mouse_folder.name].append(str(mask_file))
                else:
                    print(f"Warning: Mask file {mask_file} does not exist for scan {scan_file}.")
        
    return scan_paths, mask_paths

def get_overlapping_keys(
    bolus_scan_paths: Dict[str, List[str]], 
    molecular_scan_paths: Dict[str, List[str]]
) -> set:
    """
    Get overlapping keys between bolus and molecular scan paths.
    
    Args:
        bolus_scan_paths (Dict[str, List[str]]): Bolus scan paths.
        molecular_scan_paths (Dict[str, List[str]]): Molecular scan paths.
        
    Returns:
        set: A set of overlapping keys.
    """
    return set(bolus_scan_paths.keys()) & set(molecular_scan_paths.keys())
