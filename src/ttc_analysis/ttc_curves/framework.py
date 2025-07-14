import numpy as np
from typing import List

from ...data_objs.analysis import TtcAnalysisBase
from ...data_objs.image import UltrasoundImage
from ...data_objs.seg import CeusSeg
from .functions import *

class_name = "TtcCurvesAnalysis"

class TtcCurvesAnalysis(TtcAnalysisBase):
    """
    Class to complete RF analysis via the sliding window technique
    and generate a corresponding parametric map.
    """
    def __init__(self, image_data: UltrasoundImage, seg: CeusSeg, 
                 curve_groups: List[str], **kwargs):
        # Type checking
        assert isinstance(image_data, UltrasoundImage), 'image_data must be an UltrasoundImage child class'
        assert isinstance(seg, CeusSeg), 'seg_data must be a CeusSeg'
        super().__init__()
        
        if 'pyradiomics' in curve_groups:
            pyradiomics_config_path = kwargs['pyradiomics_config_path']
            kwargs['extractor'] = featureextractor.RadiomicsFeatureExtractor(pyradiomics_config_path)
        
        self.analysis_kwargs = kwargs
        self.curve_groups = curve_groups
        self.seg_data = seg
        self.image_data = image_data
        self.assign_curve_funcs()
        self.curves = {}  # Dictionary to hold computed curves
        
        self.time_arr = np.arange(self.image_data.pixel_data.shape[3]*self.image_data.frame_rate)
            
    def assign_curve_funcs(self):
        """
        Assign curve functions based on the provided curve names.
        """
        self.curve_funcs = {name: globals()[name]['func'] for name in self.curve_groups if name in globals()}

    def extract_frame_features(self, frame: np.ndarray, mask: np.ndarray):
        """Compute parametric map values for a single frame.
        Args:
            frame (np.ndarray): The ultrasound frame data.
            mask (np.ndarray): The mask for the region of interest.
        Returns:
            np.ndarray: The computed parametric map values for the frame.
        """
        used_curve_names = []
        for curve_group in self.curve_groups:
            curve_function = self.curve_funcs[curve_group]
            curve_names, vals = curve_function(self.image_data, frame, mask, **self.analysis_kwargs)

            for name, val in zip(curve_names, vals):
                if name in used_curve_names:
                    raise ValueError(f"Curve name '{name}' is being computed multiple times.")
                if not name in self.curves.keys():
                    self.curves[name] = []
                self.curves[name].append(val)
                used_curve_names.append(name)
