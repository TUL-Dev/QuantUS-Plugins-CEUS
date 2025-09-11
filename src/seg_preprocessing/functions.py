import numpy as np

from .decorators import required_kwargs
from ..data_objs.image import UltrasoundImage
from ..data_objs.seg import CeusSeg
from ..image_preprocessing.transforms import resample_to_spacing

@required_kwargs('target_vox_size', 'interp')
def resample(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
    """
    Resample the image data to a new spacing.

    Kwargs:
        target_vox_size: tuple of (z, y, x) spacing in mm to resample the image to.
        interp: interpolation method, one of 'nearest', 'linear', 'cubic'.
    """
    target_vox_size = kwargs['target_vox_size']
    interp = kwargs['interp']
    if 'original_spacing' in image_data.extras_dict.keys():
        pixdim = image_data.extras_dict['original_spacing']
    else:
        pixdim = image_data.pixdim

    seg_data.seg_mask = resample_to_spacing(seg_data.seg_mask, pixdim, target_vox_size, interp=interp)

    return seg_data
