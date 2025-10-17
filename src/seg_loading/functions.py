from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

from .decorators import extensions
from ..data_objs.seg import CeusSeg
from ..data_objs.image import UltrasoundImage

@extensions(".nii", ".nii.gz")
def nifti(image_data: UltrasoundImage, seg_path: str, **kwargs) -> CeusSeg:
    """
    Load ROI/VOI data from a NIfTI file. segmentation mask is used as-is.
    """
    assert seg_path.endswith('.nii.gz') or seg_path.endswith('.nii'), "seg_path must be a NIfTI file"
    
    out = CeusSeg()
    seg = nib.load(seg_path)

    # Get the number of frames from the image data and prepare the mask for it
    frame_number = image_data.pixel_data.shape[3]
    use_mc = True
    if use_mc == False:
        out.seg_mask = np.asarray(seg.dataobj, dtype=np.uint8) 
    else:
        # Load the base 3D mask
        base_mask_3d = np.asarray(seg.dataobj, dtype=np.uint8)
        out.seg_mask = np.repeat(base_mask_3d[..., np.newaxis], frame_number, axis=-1)
        print('The shape of the motion compensated mask is ' + str(out.seg_mask.shape))
        

    if out.seg_mask.ndim == 3: # 2D + time
        out.pixdim = seg.header.get_zooms()[:2]
    elif out.seg_mask.ndim == 4: # 3D + time
        out.pixdim = seg.header.get_zooms()[:3]
    
    if seg_path.endswith('.nii.gz'):
        out.seg_name = Path(seg_path).name[:-7]  # Remove '.nii.gz'
    else:
        out.seg_name = Path(seg_path).name[:-4]  # Remove '.nii'

    return out

@extensions(".nii", ".nii.gz")
def load_bolus_mask(image_data: UltrasoundImage, seg_path: str, **kwargs) -> CeusSeg:
    """
    Load a bolus mask from a given path.
    
    Args:
        mask_path (str): The path to the bolus mask file.
        
    Returns:
        sitk.Image: The loaded bolus mask image.
    """
    bolus_seg = np.asarray(nib.load(seg_path).dataobj, dtype=np.uint8).T
    bolus_seg = (np.max(bolus_seg, axis=3) > 0).astype(np.uint8)
    bolus_seg = bolus_seg[:, :image_data.pixel_data.shape[1], :]
    bolus_seg = binary_fill_holes(bolus_seg).astype(np.uint8)
    for i in range(bolus_seg.shape[0]):
        bolus_seg[i, :, :] = binary_fill_holes(bolus_seg[i, :, :])
    for i in range(bolus_seg.shape[1]):
        bolus_seg[:, i, :] = binary_fill_holes(bolus_seg[:, i, :])
    for i in range(bolus_seg.shape[2]):
        bolus_seg[:, :, i] = binary_fill_holes(bolus_seg[:, :, i])
    
    out = CeusSeg()
    out.seg_mask = bolus_seg
    out.pixdim = image_data.pixdim
    if seg_path.endswith('.nii.gz'):
        out.seg_name = Path(seg_path).name[:-7]  # Remove '.nii.gz'
    else:
        out.seg_name = Path(seg_path).name[:-4]  # Remove '.nii'

    return out