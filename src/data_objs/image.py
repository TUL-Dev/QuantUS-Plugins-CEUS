from typing import List
from pathlib import Path

import numpy as np

class UltrasoundImage:
    """
    Class for general ultrasound image data (e.g., B-mode, CEUS, NIfTI).
    """
    def __init__(self, scan_path: str):
        self.scan_name = Path(scan_path).stem
        self.pixel_data: np.ndarray # image data as a numpy array
        self.pixdim: List[float] # mm
        self.frame_rate: float # Hz
        self.intensities_for_analysis: np.ndarray # linearized intensity values
