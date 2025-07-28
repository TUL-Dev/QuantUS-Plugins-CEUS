import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from ...data_objs.image import UltrasoundImage
from ...data_objs.seg import CeusSeg
from .functions import *

class_name = "TtcCurvesAnalysis"

class TtcCurvesAnalysis:
    """
    Class to complete RF analysis via the sliding window technique
    and generate a corresponding parametric map.
    """
    def __init__(self, image_data: UltrasoundImage, seg: CeusSeg, 
                 curve_groups: List[str], **kwargs):
        # Type checking
        assert isinstance(image_data, UltrasoundImage), 'image_data must be an UltrasoundImage child class'
        assert isinstance(seg, CeusSeg), 'seg_data must be a CeusSeg'
    
        self.analysis_kwargs = kwargs
        self.curve_groups = curve_groups
        self.seg_data = seg
        self.image_data = image_data
        self.curves: Dict[str, List[float]] = {}  # Dictionary to hold computed curves
        self.curve_funcs: Dict[str, callable] = {name: globals()[name]['func'] for name in self.curve_groups if name in globals()}
        self.curves_output_path = self.analysis_kwargs.get('curves_output_path', None)
        
    def compute_curves(self):
        """Compute UTC parameters for each window in the ROI, creating a parametric map.
        """
        self.time_arr = np.arange(self.image_data.pixel_data.shape[3])*self.image_data.frame_rate

        for frame_ix, frame in tqdm(enumerate(range(self.image_data.intensities_for_analysis.shape[3])), 
                                    desc="Computing curves", total=self.image_data.intensities_for_analysis.shape[3]):
            frame_data = self.image_data.intensities_for_analysis[:, :, :, frame]
            self.extract_frame_features(frame_data, self.seg_data.seg_mask, frame_ix)

        if self.curves_output_path is not None:
            self.save_curves()

    def extract_frame_features(self, frame: np.ndarray, mask: np.ndarray, frame_ix: int):
        """Compute parametric map values for a single frame.
        Args:
            frame (np.ndarray): The ultrasound frame data.
            mask (np.ndarray): The mask for the region of interest.
            frame_ix (int): The index of the frame being processed.
        Returns:
            np.ndarray: The computed parametric map values for the frame.
        """
        used_curve_names = []
        for curve_group in self.curve_groups:
            curve_function = self.curve_funcs[curve_group]
            curve_names, vals = curve_function(self.image_data, frame, mask, **self.analysis_kwargs)

            if not len(curve_names) and not len(vals):
                if frame_ix:
                    print(self.image_data.scan_name, self.seg_data.seg_name, frame_ix, "No curves computed for this frame.")
                    for name in self.curves.keys():
                        self.curves[name].append(self.curves[name][-1])  # Append last value if no new values
            else:
                for name, val in zip(curve_names, vals):
                    if name in used_curve_names:
                        raise ValueError(f"Curve name '{name}' is being computed multiple times.")
                    if not name in self.curves.keys():
                        self.curves[name] = []
                    self.curves[name].append(val)
                    used_curve_names.append(name)
                    if frame_ix == len(self.curves[name]):
                        self.curves[name].append(val)

    def save_curves(self):
        """Save the computed curves to a CSV file.
        """
        assert type(self.curves_output_path) is str, "Export path must be a string."
        assert self.curves_output_path.endswith('.csv'), \
              "Export path must end with .csv to export to CSV format."

        Path(self.curves_output_path).parent.mkdir(parents=True, exist_ok=True)

        if not len(self.curves):
            raise ValueError("No curves have been computed.")
        
        curve_dict = {}
        curve_dict["Scan Name"] = self.image_data.scan_name
        curve_dict["Segmentation Name"] = self.seg_data.seg_name
        curve_dict["Time Array"] = self.time_arr
        for name, values in self.curves.items():
            curve_dict[name] = values

        df = pd.DataFrame(curve_dict)
        df.to_csv(self.curves_output_path, index=False)