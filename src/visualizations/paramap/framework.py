from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ...data_objs.visualizations import ParamapDrawingBase
from ...curve_quantification.framework import CurveQuantifications
from ...time_series_analysis.curves_paramap.framework import CurvesParamapAnalysis
from .functions import *

class_name = "ParamapVisualizations"

class ParamapVisualizations(ParamapDrawingBase):
    """
    Class to complete visualizations of parametric map-based UTC analysis.
    """

    def __init__(self, quants_obj: CurveQuantifications, params: List[str], custom_funcs: List[str], **kwargs):
        assert isinstance(quants_obj, CurveQuantifications), "quants_obj must be a CurveQuantifications"
        assert isinstance(quants_obj.analysis_objs, CurvesParamapAnalysis), "Can only use these visualizations with parametric map analysis"
        assert isinstance(kwargs.get('paramap_folder_path', None), str), "paramap_folder_path must be specified in kwargs"
        super().__init__()

        self.paramap_folder_path = kwargs['paramap_folder_path']
        self.hide_all_visualizations = kwargs.get('hide_all_visualizations', False)
        
        self.params = params
        self.quants_obj = quants_obj
        self.results_df = pd.DataFrame(quants_obj.data_dict)
        self.custom_funcs = custom_funcs
        self.kwargs = kwargs
        self.numerical_paramaps = []
        self.colored_paramaps = []
        self.paramap_names = []

        if not self.params:
            if hasattr(self.quants_obj.analysis_objs, 'cor_vox_len'):
                self.params = list(self.results_df.columns)[8:]
            else:
                self.params = list(self.results_df.columns)[6:]

    def draw_paramap(self, param: str, cmap: np.ndarray) -> None:
        """Draws the parametric map for the specified parameter.

        Args:
            param (str): The name of the parameter to draw.
            cmap (np.ndarray): The colormap to use for the parametric map.
        """
        assert param in self.results_df.columns, f"Parameter '{param}' not found in quantified results"
    
        numerical_paramap = np.full(self.quants_obj.analysis_objs.seg_data.seg_mask.shape, dtype=np.float32, fill_value=np.nan)
        colored_paramap = np.zeros(list(self.quants_obj.analysis_objs.seg_data.seg_mask.shape) + [4], dtype=np.uint8)
        param_vals = self.results_df[param].values

        min_val = min(param_vals); max_val = max(param_vals)
        if np.isinf(max_val):
            finite_vals = [v for v in param_vals if np.isfinite(v)]
            max_val = max(finite_vals) if finite_vals else 0
        if np.isneginf(min_val):
            finite_vals = [v for v in param_vals if np.isfinite(v)]
            min_val = min(finite_vals) if finite_vals else 0
         
        for row in self.results_df.itertuples():
            ax_start = int(row[3]); sag_start = int(row[4])
            ax_end = int(row[5]); sag_end = int(row[6])
            if hasattr(self.quants_obj.analysis_objs, 'cor_vox_len'):
                cor_start = int(row[7]); cor_end = int(row[8])
            
            num = getattr(row, param)
            if not np.isfinite(num):
                continue
            color_ix = int((255 / (max_val - min_val)) * (num - min_val)) if min_val != max_val else 125
            if hasattr(self.quants_obj.analysis_objs, 'cor_vox_len'):
                numerical_paramap[sag_start:sag_end+1, cor_start:cor_end+1, ax_start:ax_end+1] = num
                colored_paramap[sag_start:sag_end+1, cor_start:cor_end+1, ax_start:ax_end+1, :3] = (np.array(cmap[color_ix])*255).astype(np.uint8)
                colored_paramap[sag_start:sag_end+1, cor_start:cor_end+1, ax_start:ax_end+1, 3] = 255
            else:
                numerical_paramap[ax_start:ax_end+1, sag_start:sag_end+1] = num
                colored_paramap[ax_start:ax_end+1, sag_start:sag_end+1, :3] = (np.array(cmap[color_ix])*255).astype(np.uint8)
                colored_paramap[ax_start:ax_end+1, sag_start:sag_end+1, 3] = 255
        
        # Trim parametric map to the segmentation mask
        mask_bkgr = np.where(self.quants_obj.analysis_objs.seg_data.seg_mask == 0)
        colored_paramap[mask_bkgr] = 0
        numerical_paramap[mask_bkgr] = np.nan

        return colored_paramap, numerical_paramap


    def save_paramap(self, paramap: np.ndarray, dest_path: Path) -> None:
        """Saves the parametric map to the specified path.
        
        Args:
            paramap (np.ndarray): The parametric map to save.
            dest_path (str): The destination path for saving the parametric map.
        """
        assert str(dest_path).endswith('.npy'), "Parametric map output path must end with .npy"

        np.save(dest_path, paramap)

    def generate_visualizations(self):
        """Generate visualizations for the parametric maps."""
        assert len(self.params), "No parameters to visualize"

        # Generate parametric maps
        for cmap_ix, param in enumerate(self.params):
            colored_paramap, numerical_paramap = self.draw_paramap(param, self.cmaps[cmap_ix % len(self.cmaps)])
            self.numerical_paramaps.append(numerical_paramap)
            self.colored_paramaps.append(colored_paramap)
            self.paramap_names.append(param)
                       
        # Complete all custom visualizations
        for func_name in self.custom_funcs:
            function = globals()[func_name]
            function(self.quants_obj, **self.kwargs)

        if not self.hide_all_visualizations:
            self.export_visualizations()

    def export_visualizations(self):
        """Used to specify which visualizations to export and where.
        """
        paramap_folder_path = Path(self.paramap_folder_path)
        paramap_folder_path.mkdir(parents=True, exist_ok=True)
        
        im = self.quants_obj.analysis_objs.image_data.pixel_data
        seg = self.quants_obj.analysis_objs.seg_data.seg_mask
        np.save(paramap_folder_path / 'image.npy', im)
        np.save(paramap_folder_path / 'segmentation.npy', seg)
        np.save(paramap_folder_path / 'pix_dims.npy', self.quants_obj.analysis_objs.image_data.pixdim)

        # Save parametric maps
        for numerical_paramap, colored_paramap, param in zip(self.numerical_paramaps, self.colored_paramaps, self.paramap_names):
            self.save_paramap(colored_paramap, paramap_folder_path / f'{param}_colored.npy')
            self.save_paramap(numerical_paramap, paramap_folder_path / f'{param}_numerical.npy')
