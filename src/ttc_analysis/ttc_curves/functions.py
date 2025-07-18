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

    image = sitk.GetImageFromArray(frame.T)
    image.SetSpacing(tuple(float(x) for x in image_data.pixdim))
    mask_im = sitk.GetImageFromArray(mask.T)
    mask_im.SetSpacing(tuple(float(x) for x in image_data.pixdim))

    for config_path in kwargs['pyradiomics_config_paths']:
        if not isinstance(config_path, str):
            raise ValueError("Each config path must be a string.")
        extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
        features = extractor.execute(image, mask_im)

        feature_names = []; feature_vals = []
        for name in features.keys():
            val_type = type(features[name])
            if val_type == list or val_type == dict or val_type == tuple or val_type == str:
                continue
            feature_names.append(f"{Path(config_path).stem}_{name}"); feature_vals.append(features[name])

    return feature_names, feature_vals
