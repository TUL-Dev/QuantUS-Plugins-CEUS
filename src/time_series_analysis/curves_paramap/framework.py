import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from ...data_objs.image import UltrasoundImage
from ...data_objs.seg import CeusSeg
from ..curve_types.functions import *
from ..curves.framework import CurvesAnalysis

class_name = "CurvesParamap"

class CurvesParamapAnalysis(CurvesAnalysis):
    """
    Class to complete RF analysis via the sliding window technique
    and generate a corresponding parametric map.
    """
    required_kwargs = ['ax_vox_len', 'sag_vox_len', 'cor_vox_len', 'ax_vox_ovrlp', 'sag_vox_ovrlp', 'cor_vox_ovrlp']
    
    def __init__(self, image_data: UltrasoundImage, seg: CeusSeg, 
                 curve_groups: List[str], **kwargs):
        super().__init__(image_data, seg, curve_groups, **kwargs)
        assert 'ax_vox_len' in kwargs.keys(), 'Must include axial voxel length for parametric map computation'
        assert 'sag_vox_len' in kwargs.keys(), 'Must include sagittal voxel length for parametric map computation'
        assert 'ax_vox_ovrlp' in kwargs.keys(), 'Must include axial voxel overlap for parametric map computation'
        assert 'sag_vox_ovrlp' in kwargs.keys(), 'Must include sagittal voxel length for parametric map computation'
        assert 'cor_vox_len' in kwargs.keys() or image_data.intensities_for_analysis.ndim == 3, 'Must include coronal voxel length for parametric map computation'
        assert 'cor_vox_ovrlp' in kwargs.keys() or image_data.intensities_for_analysis.ndim == 3, 'Must include coronal voxel overlap for parametric map computation'

        if image_data.intensities_for_analysis.ndim == 4:
            self.cor_vox_len = kwargs['cor_vox_len']        # mm
            self.cor_vox_ovrlp = kwargs['cor_vox_ovrlp']    # %
            self.cor_res = self.image_data.pixdim[2]        # mm/px

        self.ax_vox_len = kwargs['ax_vox_len']              # mm
        self.sag_vox_len = kwargs['sag_vox_len']            # mm
        self.ax_vox_ovrlp = kwargs['ax_vox_ovrlp']          # %
        self.sag_vox_ovrlp = kwargs['sag_vox_ovrlp']        # %
        self.ax_res = self.image_data.pixdim[0]             # mm/px
        self.sag_res = self.image_data.pixdim[1]            # mm/px

        self.windows = self.generate_windows()              # Generate sliding windows for the parametric map
        self.curves: List[Dict[str, List[float]]] = []  # List to hold computed curves for each window

    def generate_windows(self):
        """Generate sliding windows for the parametric map.

        Returns:
            List[tuple]: List of tuples containing window coordinates.
        """
        windows = []
        ax_step = int(self.ax_vox_len * (1 - (self.ax_vox_ovrlp / 100)) / self.ax_res)
        sag_step = int(self.sag_vox_len * (1 - (self.sag_vox_ovrlp / 100)) / self.sag_res)
        if hasattr(self, 'cor_vox_len'):
            cor_step = int(self.cor_vox_len * (1 - (self.cor_vox_ovrlp / 100)) / self.cor_res)

        seg_mask = np.asarray(self.seg_data.seg_mask)
        seg_mask_bool = seg_mask.astype(bool)
        if (
            self.image_data.intensities_for_analysis.ndim == 3
            and seg_mask.ndim == 3
            and not hasattr(self, 'cor_vox_len')
        ):
            # Motion-compensated segmentations provide a mask per frame; collapse across time.
            seg_mask_for_windows = np.any(seg_mask_bool, axis=0)
        else:
            seg_mask_for_windows = seg_mask_bool

        # Create minimum and maximum indices for the sliding windows based on the segmentation mask
        mask_ixs = np.where(seg_mask_for_windows)
        if hasattr(self, 'cor_vox_len'):
            min_ax, max_ax = np.min(mask_ixs[2]), np.max(mask_ixs[2])
            min_sag, max_sag = np.min(mask_ixs[0]), np.max(mask_ixs[0])
            min_cor, max_cor = np.min(mask_ixs[1]), np.max(mask_ixs[1])
        else:
            min_ax, max_ax = np.min(mask_ixs[0]), np.max(mask_ixs[0])
            min_sag, max_sag = np.min(mask_ixs[1]), np.max(mask_ixs[1])

        for ax_start in range(min_ax, max_ax, ax_step):
            for sag_start in range(min_sag, max_sag, sag_step):
                if hasattr(self, 'cor_vox_len'):
                    for cor_start in range(min_cor, max_cor, cor_step):
                            # Determine if window is inside analysis volume
                            mask_vals = seg_mask_for_windows[
                                sag_start : (sag_start + sag_step),
                                cor_start : (cor_start + cor_step),
                                ax_start : (ax_start + ax_step),
                            ]
                            
                            # Define Percentage Threshold
                            total_number_of_elements_in_region = mask_vals.size
                            number_of_ones_in_region = np.count_nonzero(mask_vals)
                            percentage_ones = number_of_ones_in_region / total_number_of_elements_in_region

                            if percentage_ones > 0.2:
                                windows.append((ax_start, sag_start, cor_start, 
                                                ax_start + ax_step, sag_start + sag_step, cor_start + cor_step))
                else:
                    # Determine if window is inside analysis volume
                    mask_vals = seg_mask_for_windows[
                        ax_start : (ax_start + ax_step),
                        sag_start : (sag_start + sag_step),
                    ]
                    
                    # Define Percentage Threshold
                    total_number_of_elements_in_region = mask_vals.size
                    number_of_ones_in_region = np.count_nonzero(mask_vals)
                    percentage_ones = number_of_ones_in_region / total_number_of_elements_in_region

                    if percentage_ones > 0.2:
                        windows.append((ax_start, sag_start, 
                                        ax_start + ax_step, sag_start + sag_step))

        return windows
        
    def compute_curves(self):
        """Compute UTC parameters for each window in the ROI, creating a parametric map."""
        if self.image_data.intensities_for_analysis.ndim == 4: # 3D + time
            for frame_ix, frame in tqdm(enumerate(range(self.image_data.intensities_for_analysis.shape[3])), 
                                        desc="Computing curves", total=self.image_data.intensities_for_analysis.shape[3]):
                frame_data = self.image_data.intensities_for_analysis[:, :, :, frame]
                for window_ix, window in enumerate(self.windows):
                    mask = np.zeros_like(self.seg_data.seg_mask)
                    ax_start, sag_start, cor_start, ax_end, sag_end, cor_end = window
                    mask[sag_start:sag_end+1, 
                            cor_start:cor_end+1, 
                            ax_start:ax_end+1] = 1
                    self.extract_frame_features(frame_data, mask, frame_ix, window_ix)
        elif self.image_data.intensities_for_analysis.ndim == 3: # 2D + time
            is_mask_3d = len(self.seg_data.seg_mask.shape) == 3 

            for frame_ix, frame in tqdm(enumerate(range(self.image_data.intensities_for_analysis.shape[0])), 
                                        desc="Computing curves", total=self.image_data.intensities_for_analysis.shape[0]):
                frame_data = self.image_data.intensities_for_analysis[frame_ix]
                for window_ix, window in enumerate(self.windows):
                    if is_mask_3d:
                        # print(self.seg_data.seg_mask[frame_ix,:,:].shape)    
                        mask = np.zeros_like(self.seg_data.seg_mask[frame_ix])
                        ax_start, sag_start, ax_end, sag_end = window 
                        mask[ax_start:ax_end+1, sag_start:sag_end+1] = 1
                        self.extract_frame_features(frame_data, mask, frame_ix, window_ix)
                    else:
                        mask = np.zeros_like(self.seg_data.seg_mask)
                        ax_start, sag_start, ax_end, sag_end = window
                        mask[ax_start:ax_end+1, sag_start:sag_end+1] = 1
                        self.extract_frame_features(frame_data, mask, frame_ix, window_ix)
        else:
            raise ValueError("Image data must be either 2D+time or 3D+time.")

        if self.curves_output_path:
            self.save_curves()

    def extract_frame_features(self, frame: np.ndarray, mask: np.ndarray, frame_ix: int, window_ix: int):
        """Compute parametric map values for a single frame."""
        if len(self.curves) < window_ix:
            raise ValueError(f"Window index {window_ix} exceeds the number of computed curves.")
        elif len(self.curves) == window_ix:
            self.curves.append({})
            window = self.windows[window_ix]
            self.curves[window_ix]['Window-Axial Start Pix'] = window[0]
            self.curves[window_ix]['Window-Sagittal Start Pix'] = window[1]
            if hasattr(self, 'cor_vox_len'):
                self.curves[window_ix]['Window-Coronal Start Pix'] = window[2]
                self.curves[window_ix]['Window-Axial End Pix'] = window[3]
                self.curves[window_ix]['Window-Sagittal End Pix'] = window[4]
                self.curves[window_ix]['Window-Coronal End Pix'] = window[5]
            else:
                self.curves[window_ix]['Window-Axial End Pix'] = window[2]
                self.curves[window_ix]['Window-Sagittal End Pix'] = window[3]

        for curve_group in self.curve_groups:
            curve_function = self.curve_funcs[curve_group]
            curve_names, vals = curve_function(self.image_data, frame, mask, **self.analysis_kwargs)

            for name, val in zip(curve_names, vals):
                if name not in self.curves[window_ix]:
                    self.curves[window_ix][name] = []
                    self.curves[window_ix][name].append(val)
                elif frame_ix > 0:
                    self.curves[window_ix][name].append(val)

    def verify_motion_compensated_windows(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        coverage_threshold: float = 0.2,
        display_inline: bool = True,
        show_window_index: bool = True,
        save_path: Optional[Path] = None,
        fps: int = 15,
        pause_time: float = 0.05,
        figure_size: Tuple[int, int] = (10, 6),
    ) -> pd.DataFrame:
        """Verify and visualize motion-compensated window indexing for 2D+time datasets.

        For each frame, overlays the segmentation boundary and sliding-window bounding boxes on
        the motion-compensated frame. Windows that fall below the specified coverage threshold or
        extend outside the frame are highlighted in red. Optionally displays the animation inline
        (e.g., in a notebook) and/or saves the frames as a video (GIF/MP4) via imageio.

        Args:
            start_frame: First frame index to include.
            end_frame: Last frame index (exclusive). Defaults to the final frame.
            coverage_threshold: Minimum fraction of mask pixels within a window to consider it valid.
            display_inline: If True, streams the animation inline using IPython display utilities.
            show_window_index: If True, annotate each window with its index.
            save_path: Optional path to write a GIF/MP4. Requires imageio if provided.
            fps: Frames per second when writing a video/GIF.
            pause_time: Delay between frames when display_inline is True.
            figure_size: Matplotlib figure size.

        Returns:
            pd.DataFrame summarizing per-frame window coverage statistics.

        Raises:
            NotImplementedError: If executed on 3D+time data.
        """

        if self.image_data.intensities_for_analysis.ndim != 3:
            raise NotImplementedError(
                "Motion-compensated window verification currently supports only 2D + time datasets."
            )

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from scipy.ndimage import binary_erosion

        ipy_display = None
        ipy_clear = None
        if display_inline:
            try:
                from IPython.display import display as ipy_display  # type: ignore
                from IPython.display import clear_output as ipy_clear  # type: ignore
            except ImportError:
                ipy_display = None
                ipy_clear = None

        frame_data = self.image_data.intensities_for_analysis
        mask_data = self.seg_data.seg_mask
        is_mask_time_resolved = mask_data.ndim == 3

        frame_count = frame_data.shape[0]
        if end_frame is None or end_frame > frame_count:
            end_frame = frame_count
        start_frame = max(start_frame, 0)
        frame_indices = range(start_frame, end_frame)

        def get_mask_boundary(mask_slice: np.ndarray) -> np.ndarray:
            if mask_slice.size == 0 or mask_slice.max() == 0:
                return np.zeros_like(mask_slice, dtype=bool)
            eroded = binary_erosion(mask_slice)
            return mask_slice.astype(bool) & ~eroded

        window_stats: List[Dict[str, float]] = []
        captured_frames: List[np.ndarray] = []

        for frame_ix in frame_indices:
            frame_img = np.asarray(frame_data[frame_ix])
            mask_slice = np.asarray(mask_data[frame_ix] if is_mask_time_resolved else mask_data)

            boundary = get_mask_boundary(mask_slice)

            fig, ax = plt.subplots(figsize=figure_size)
            ax.imshow(np.asarray(frame_img), cmap="gray")
            ax.set_title(f"Frame {frame_ix}")
            ax.set_xlabel("Sagittal (columns)")
            ax.set_ylabel("Axial (rows)")

            if mask_slice.ndim >= 2:
                ax.contour(boundary, colors="red", linewidths=1.2)

            frame_height, frame_width = frame_img.shape[-2], frame_img.shape[-1]

            for window_ix, window in enumerate(self.windows):
                ax_start, sag_start, ax_end, sag_end = window
                ax_end_inclusive = ax_end
                sag_end_inclusive = sag_end
                row_slice = slice(ax_start, min(ax_end_inclusive + 1, frame_height))
                col_slice = slice(sag_start, min(sag_end_inclusive + 1, frame_width))

                out_of_bounds = (
                    ax_start < 0
                    or sag_start < 0
                    or ax_end_inclusive >= frame_height
                    or sag_end_inclusive >= frame_width
                )

                window_mask = mask_slice[row_slice, col_slice]
                window_area = window_mask.size if window_mask.size else (ax_end - ax_start + 1) * (sag_end - sag_start + 1)
                mask_pixels = int(np.count_nonzero(window_mask)) if window_mask.size else 0
                coverage = mask_pixels / window_area if window_area else 0.0

                window_stats.append(
                    {
                        "frame_index": frame_ix,
                        "window_index": window_ix,
                        "coverage_fraction": coverage,
                        "mask_pixels_in_window": mask_pixels,
                        "window_pixels": window_area,
                        "out_of_bounds": out_of_bounds,
                    }
                )

                rect_width = sag_end_inclusive - sag_start + 1
                rect_height = ax_end_inclusive - ax_start + 1
                rect_color = "lime" if (coverage >= coverage_threshold and not out_of_bounds) else "red"
                rect = Rectangle(
                    (sag_start, ax_start),
                    rect_width,
                    rect_height,
                    linewidth=1.5,
                    edgecolor=rect_color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                if show_window_index:
                    ax.text(
                        sag_start + rect_width / 2,
                        ax_start + rect_height / 2,
                        str(window_ix),
                        color=rect_color,
                        ha="center",
                        va="center",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.3),
                    )

            fig.tight_layout()

            if save_path is not None:
                fig.canvas.draw()
                if hasattr(fig.canvas, "buffer_rgba"):
                    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    frame_rgb = rgba[..., :3]
                else:
                    frame_rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame_rgb = frame_rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                captured_frames.append(frame_rgb.copy())

            if ipy_display and ipy_clear:
                ipy_clear(wait=True)
                ipy_display(fig)
                plt.pause(0.001)

            plt.close(fig)

        # if save_path is not None and captured_frames:
        #     try:
        #         import imageio.v3 as iio

        #         output_ext = Path(save_path).suffix.lower()
        #         if output_ext in {".gif"}:
        #             iio.imwrite(save_path, captured_frames, duration=1 / max(fps, 1))
        #         else:
        #             iio.imwrite(save_path, captured_frames, fps=fps)
        #     except ImportError as exc:
        #         raise ImportError(
        #             "Saving the verification animation requires imageio. Install it or omit save_path."
        #         ) from exc

        stats_df = pd.DataFrame(window_stats)
        return stats_df
