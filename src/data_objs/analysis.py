from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
from typing import Dict, List

from .seg import CeusSeg
from .image import UltrasoundImage
        
class TtcAnalysisBase(ABC):
    """Facilitate parametric map-centric analysis of ultrasound images.
    """
    
    def __init__(self):
        self.image_data: UltrasoundImage
        self.seg_data: CeusSeg
        
        self.time_arr: np.ndarray
        self.curves: Dict[str, List[float]]
        self.curve_funcs: Dict[str, callable]
                    
    @abstractmethod
    def extract_frame_features(self, frame: np.ndarray, mask: np.ndarray):
        """Compute parametric map values for a single frame.
        Args:
            frame (np.ndarray): The ultrasound frame data.
            mask (np.ndarray): The mask for the region of interest.
        Returns:
            np.ndarray: The computed parametric map values for the frame.
        """
        pass

    def compute_curves(self):
        """Compute UTC parameters for each window in the ROI, creating a parametric map.
        """
        for frame in tqdm(range(self.image_data.intensities_for_analysis.shape[3]), desc="Computing curves"):
            frame_data = self.image_data.intensities_for_analysis[:, :, :, frame]
            self.extract_frame_features(frame_data, self.seg_data.seg_mask)
