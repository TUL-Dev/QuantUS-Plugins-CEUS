"""
3D Motion Compensation for CEUS Analysis
Based on ILSA tracking, directly creates mc_seg_mask
"""

import numpy as np
from scipy.ndimage import shift
from scipy.signal import correlate
from typing import Tuple, List, Optional
from dataclasses import dataclass
from collections import Counter

@dataclass
class BoundingBox3D:
    """3D Bounding box definition (z, y, x) format"""
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.z_max - self.z_min, self.y_max - self.y_min, self.x_max - self.x_min)
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.z_min + self.z_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.x_min + self.x_max) / 2
        )
    
    def expand(self, margin: Tuple[int, int, int]) -> 'BoundingBox3D':
        return BoundingBox3D(
            max(0, self.z_min - margin[0]), self.z_max + margin[0],
            max(0, self.y_min - margin[1]), self.y_max + margin[1],
            max(0, self.x_min - margin[2]), self.x_max + margin[2]
        )
    
    def extract_from_volume(self, volume: np.ndarray) -> np.ndarray:
        return volume[self.z_min:self.z_max, self.y_min:self.y_max, self.x_min:self.x_max]
    
    def translate(self, dz: int, dy: int, dx: int) -> 'BoundingBox3D':
        return BoundingBox3D(
            self.z_min + dz, self.z_max + dz,
            self.y_min + dy, self.y_max + dy,
            self.x_min + dx, self.x_max + dx
        )
    
    @classmethod
    def from_mask(cls, mask: np.ndarray, padding: int = 0) -> 'BoundingBox3D':
        """Create bounding box from binary mask (z, y, x)"""
        nonzero = np.argwhere(mask > 0)
        if len(nonzero) == 0:
            raise ValueError("Mask contains no non-zero voxels")
        
        z_min = max(0, nonzero[:, 0].min() - padding)
        z_max = min(mask.shape[0], nonzero[:, 0].max() + 1 + padding)
        y_min = max(0, nonzero[:, 1].min() - padding)
        y_max = min(mask.shape[1], nonzero[:, 1].max() + 1 + padding)
        x_min = max(0, nonzero[:, 2].min() - padding)
        x_max = min(mask.shape[2], nonzero[:, 2].max() + 1 + padding)
        
        return cls(z_min, z_max, y_min, y_max, x_min, x_max)

class MotionCompensation3D:
    """3D Motion Compensation using ILSA tracking"""
    
    def __init__(self, search_margin_ratio: float = 0.5 / 30):
        self.search_margin_ratio = search_margin_ratio
    
    def compute_search_margin(self, volume_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        # return int(self.search_margin_ratio * volume_shape)
        return tuple(int(self.search_margin_ratio*x) for x in volume_shape)
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        mean = np.mean(volume)
        std = np.std(volume)
        if std == 0:
            return volume - mean
        return (volume - mean) / std
       
    def compute_3d_correlation_vectorized(
        self,
        volumes: np.ndarray,  # Shape: (n_frames, depth, height, width)
        reference_voi: np.ndarray,  # Shape: (ref_depth, ref_height, ref_width)
        search_bbox: BoundingBox3D
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized 3D correlation computation across all frames
        
        Args:
            volumes: Full volume data for all frames
            reference_voi: Reference volume of interest (VOI)
            search_bbox: Bounding box defining search region
            
        Returns:
            correlation_map: Shape (n_frames, search_d, search_h, search_w)
            max_correlations: Shape (n_frames,) - maximum correlation per frame
        """
        n_frames = volumes.shape[-1]
        ref_shape = reference_voi.shape
        
        # Normalize reference VOI
        ref_normalized = self.normalize_volume(reference_voi)
        
        # Initialize correlation map
        search_shape = search_bbox.shape
        # Calculate output shape after correlation
        corr_shape = tuple(s - r + 1 for s, r in zip(search_shape, ref_shape))
        correlation_map = np.zeros((n_frames, *corr_shape))
        
        # Process each frame
        for frame_idx in range(n_frames):
            # Extract search region for this frame
            search_region = search_bbox.extract_from_volume(volumes[...,frame_idx])
            
            # Normalize search region
            search_normalized = self.normalize_volume(search_region)
            
            # Compute normalized cross-correlation
            # This is equivalent to cv2.matchTemplate with TM_CCOEFF_NORMED
            correlation = correlate(
                search_normalized,
                ref_normalized,
                mode='valid',
                method='fft'
            )
            
            # Normalize correlation
            ref_sum_sq = np.sum(ref_normalized ** 2)
            
            # For each position in correlation map, compute local sum of squares
            # This is a sliding window operation
            search_sq = search_normalized ** 2
            
            # Use separable convolution for efficiency
            kernel = np.ones(ref_shape)
            local_sum_sq = correlate(search_sq, kernel, mode='valid', method='fft')
            
            # Avoid division by zero
            denominator = np.sqrt(ref_sum_sq * local_sum_sq)
            denominator = np.where(denominator == 0, 1e-10, denominator)
            
            correlation_map[frame_idx] = correlation / denominator
        
        # Find maximum correlation for each frame
        max_correlations = np.max(
            correlation_map.reshape(n_frames, -1),
            axis=1
        )
        return correlation_map, max_correlations
    
    def find_optimal_translation(
        self,
        correlation_map: np.ndarray,
        search_bbox: BoundingBox3D,
        reference_bbox: BoundingBox3D
    ) -> Tuple[int, int, int]:
        """Find optimal translation from correlation map"""
        max_idx = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        
        dz = search_bbox.z_min + max_idx[0] - reference_bbox.z_min
        dy = search_bbox.y_min + max_idx[1] - reference_bbox.y_min
        dx = search_bbox.x_min + max_idx[2] - reference_bbox.x_min
        
        return dz, dy, dx
    
    def track_motion_ilsa_3d(
        self,
        volumes: np.ndarray,
        reference_frame_idx: int,
        reference_bbox: BoundingBox3D
    ) -> Tuple[List[BoundingBox3D], List[float]]:
        """
        ILSA tracking: bidirectional with reference vs temporal neighbor selection
        """
        n_frames = volumes.shape[-1]
        ref_voi = reference_bbox.extract_from_volume(volumes[...,reference_frame_idx])
        search_margin = self.compute_search_margin(volumes.shape[:-1])
        
        tracked_bboxes = [None] * n_frames
        correlations = [0.0] * n_frames
        tracking_sources = [''] * n_frames
        
        tracked_bboxes[reference_frame_idx] = reference_bbox
        correlations[reference_frame_idx] = 1.0
        tracking_sources[reference_frame_idx] = 'reference'
        
        print("Tracking forward...")
        # Forward tracking
        for frame_idx in range(reference_frame_idx + 1, n_frames):
            prev_bbox = tracked_bboxes[frame_idx - 1]
            search_bbox = prev_bbox.expand(search_margin)
            
            # Try reference frame
            corr_map_ref, max_corr_ref = self.compute_3d_correlation_vectorized(
                volumes[...,frame_idx:frame_idx+1], ref_voi, search_bbox
            )
            
            # Try previous frame
            prev_voi = prev_bbox.extract_from_volume(volumes[...,frame_idx - 1])
            corr_map_prev, max_corr_prev = self.compute_3d_correlation_vectorized(
                volumes[...,frame_idx:frame_idx+1], prev_voi, search_bbox
            )
            
            # Pick better correlation
            if max_corr_ref[0] >= max_corr_prev[0]:
                dz, dy, dx = self.find_optimal_translation(corr_map_ref[0], search_bbox, reference_bbox)
                tracked_bboxes[frame_idx] = reference_bbox.translate(dz, dy, dx)
                correlations[frame_idx] = max_corr_ref[0]
                tracking_sources[frame_idx] = 'reference'
            else:
                dz, dy, dx = self.find_optimal_translation(corr_map_prev[0], search_bbox, prev_bbox)
                tracked_bboxes[frame_idx] = prev_bbox.translate(dz, dy, dx)
                correlations[frame_idx] = max_corr_prev[0]
                tracking_sources[frame_idx] = 'previous'
        
        print("Tracking backward...")
        # Backward tracking
        for frame_idx in range(reference_frame_idx - 1, -1, -1):
            next_bbox = tracked_bboxes[frame_idx + 1]
            search_bbox = next_bbox.expand(search_margin)
            
            # Try reference frame
            corr_map_ref, max_corr_ref = self.compute_3d_correlation_vectorized(
                volumes[...,frame_idx:frame_idx+1], ref_voi, search_bbox
            )
            
            # Try next frame
            next_voi = next_bbox.extract_from_volume(volumes[..., frame_idx + 1])
            corr_map_next, max_corr_next = self.compute_3d_correlation_vectorized(
                volumes[...,frame_idx:frame_idx+1], next_voi, search_bbox
            )
            
            # Pick better correlation
            if max_corr_ref[0] >= max_corr_next[0]:
                dz, dy, dx = self.find_optimal_translation(corr_map_ref[0], search_bbox, reference_bbox)
                tracked_bboxes[frame_idx] = reference_bbox.translate(dz, dy, dx)
                correlations[frame_idx] = max_corr_ref[0]
                tracking_sources[frame_idx] = 'reference'
            else:
                dz, dy, dx = self.find_optimal_translation(corr_map_next[0], search_bbox, next_bbox)
                tracked_bboxes[frame_idx] = next_bbox.translate(dz, dy, dx)
                correlations[frame_idx] = max_corr_next[0]
                tracking_sources[frame_idx] = 'next'
        
        # Print stats
        source_counts = Counter(tracking_sources)
        print(f"\nTracking complete!")
        print(f"  Sources: {dict(source_counts)}")
        print(f"  Mean correlation: {np.mean(correlations):.3f}")
        print(f"  Min correlation: {np.min(correlations):.3f}")
        
        return tracked_bboxes, correlations