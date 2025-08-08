import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from typing import Tuple, List

from radiomics import featureextractor

from ...data_objs.image import UltrasoundImage
from .decorators import required_kwargs

logging.getLogger('radiomics').setLevel(logging.ERROR)

@required_kwargs('pyradiomics_config_paths')
def pyradiomics(image_data: UltrasoundImage, frame: np.ndarray, mask: np.ndarray, **kwargs) -> Tuple[List[str], List[np.ndarray]]:
    """
    Extract features using Pyradiomics.
    
    Args:
        frame (np.ndarray): The ultrasound RF frame data.
        mask (np.ndarray): The mask for the region of interest.
        **kwargs: Additional keyword arguments for Pyradiomics feature extraction.
        
    Returns:
        Tuple[List[str], List[np.ndarray]]: A tuple containing the feature names and their corresponding values.
    """
    assert(type(kwargs['pyradiomics_config_paths']) == list), "pyradiomics_config_paths must be a list of paths to PyRadiomics YAML configuration files."

    # Define the affine transformation matrix
    affine = np.eye(frame.ndim + 1)
    reversed_dims = list(reversed(image_data.pixdim))
    for i in range(frame.ndim):
        affine[i, i] = reversed_dims[i]
    origin = affine[:frame.ndim, -1].tolist()
    direction_matrix = affine[:frame.ndim, :frame.ndim]
    spacing = np.linalg.norm(direction_matrix, axis=0).tolist()
    direction = (direction_matrix / spacing).flatten().tolist()

    image = sitk.GetImageFromArray(frame.T)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)
    mask_im = sitk.GetImageFromArray(mask.T)
    mask_im.SetSpacing(spacing)
    mask_im.SetOrigin(origin)
    mask_im.SetDirection(direction)

    feature_names = []; feature_vals = []
    for config_path in kwargs['pyradiomics_config_paths']:
        if not isinstance(config_path, str):
            raise ValueError("Each config path must be a string.")
        try:
            extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
            features = extractor.execute(image, mask_im)
        except Exception as e:
            logging.error(f"Error occurred while extracting features: {e}")
            continue

        for name in features.keys():
            val_type = type(features[name])
            if val_type == list or val_type == dict or val_type == tuple or val_type == str:
                continue
            feature_names.append(f"{Path(config_path).stem}_{name}"); feature_vals.append(features[name])

    return feature_names, feature_vals

@required_kwargs()
def tic(image_data: UltrasoundImage, frame: np.ndarray, mask: np.ndarray, **kwargs) -> Tuple[List[str], List[np.ndarray]]:
    """
    Extract Time Intensity Curve (TIC) features from the ultrasound image data.
    
    Args:
        image_data (UltrasoundImage): The ultrasound image data object.
        frame (np.ndarray): The ultrasound RF frame data.
        mask (np.ndarray): The mask for the region of interest.
        **kwargs: Additional keyword arguments (not used).
        
    Returns:
        Tuple[List[str], List[np.ndarray]]: A tuple containing the feature names and their corresponding values.
    """
    assert isinstance(image_data, UltrasoundImage), "image_data must be an instance of UltrasoundImage"
    
    tic_curve = np.mean(frame[mask > 0], axis=0)
    return ['TIC'], [tic_curve]