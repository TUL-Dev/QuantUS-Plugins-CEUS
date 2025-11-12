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
import cv2

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
        
        # Set reference frame
        tracked_bboxes[reference_frame_idx] = reference_bbox
        correlations[reference_frame_idx] = 1.0
        tracking_sources[reference_frame_idx] = 'reference'
        
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
            
            # Pick whichever has better correlation
            # if max_corr_ref[0] >= max_corr_prev[0]:
            dz, dy, dx = self.find_optimal_translation(
                corr_map_ref[0], search_bbox, reference_bbox
            )
            tracked_bboxes[frame_idx] = reference_bbox.translate(dz, dy, dx)
            correlations[frame_idx] = max_corr_ref[0]
            tracking_sources[frame_idx] = 'reference'
            # else:
            #     dz, dy, dx = self.find_optimal_translation(
            #         corr_map_prev[0], search_bbox, prev_bbox
            #     )
            #     tracked_bboxes[frame_idx] = prev_bbox.translate(dz, dy, dx)
            #     correlations[frame_idx] = max_corr_prev[0]
            #     tracking_sources[frame_idx] = 'previous'
        
        # print("Tracking backward...")
        # # === BACKWARD TRACKING ===
        # for frame_idx in range(reference_frame_idx - 1, -1, -1):
        #     next_bbox = tracked_bboxes[frame_idx + 1]
        #     search_bbox = next_bbox.expand(search_margin)
            
        #     # Try reference frame
        #     corr_map_ref, max_corr_ref = self.compute_3d_correlation_vectorized(
        #         volumes[...,frame_idx:frame_idx+1], ref_voi, search_bbox
        #     )
            
        #     # Try next frame
        #     next_voi = next_bbox.extract_from_volume(volumes[..., frame_idx + 1])
        #     corr_map_next, max_corr_next = self.compute_3d_correlation_vectorized(
        #         volumes[...,frame_idx:frame_idx+1], next_voi, search_bbox
        #     )
            
        #     # Pick whichever has better correlation
        #     if max_corr_ref[0] >= max_corr_next[0]:
        #         dz, dy, dx = self.find_optimal_translation(
        #             corr_map_ref[0], search_bbox, reference_bbox
        #         )
        #         tracked_bboxes[frame_idx] = reference_bbox.translate(dz, dy, dx)
        #         correlations[frame_idx] = max_corr_ref[0]
        #         tracking_sources[frame_idx] = 'reference'
        #     else:
        #         dz, dy, dx = self.find_optimal_translation(
        #             corr_map_next[0], search_bbox, next_bbox
        #         )
        #         tracked_bboxes[frame_idx] = next_bbox.translate(dz, dy, dx)
        #         correlations[frame_idx] = max_corr_next[0]
        #         tracking_sources[frame_idx] = 'next'
        
        # === PRINT STATS ===
        source_counts = Counter(tracking_sources)
        print(f"\nTracking complete!")
        print(f"  Sources: {dict(source_counts)}")
        print(f"  Mean correlation: {np.mean(correlations):.3f}")
        print(f"  Min correlation: {np.min(correlations):.3f}")
        
        return tracked_bboxes, correlations

class OpticalFlowMotionCompensation3D:
    """
    3D Motion Compensation using TRUE 3D Feature Tracking
    
    This implementation tracks features in full 3D using volumetric correlation:
    1. Detect good features to track in the reference frame (across multiple slices)
    2. For each feature, extract a 3D patch around it
    3. Search in 3D neighborhood in next frame using normalized cross-correlation
    4. Track motion in all three dimensions: (dz, dy, dx)
    5. Estimate global 3D translation from all tracked features
    
    Key advantage: Captures true 3D motion including Z-axis (through-plane) displacement
    
    Note: Unlike 2D ultrasound, we do NOT reject frames for out-of-plane motion
    since we have full 3D volumes. All frames can be tracked and compensated.
    """
    
    def __init__(
        self,
        feature_params: Optional[dict] = None,
        min_features_ratio: float = 0.2,
        patch_size_z: int = 5,
        patch_size_y: int = 7,
        patch_size_x: int = 7,
        search_range_z: int = 3,
        search_range_y: int = 7,
        search_range_x: int = 7,
        correlation_threshold: float = 0.5
    ):
        """
        Initialize optical flow motion compensation
        
        Args:
            feature_params: Parameters for good features to track
            min_features_ratio: Minimum ratio of tracked features (for confidence)
            patch_size_z: Half-size of patch in Z direction
            patch_size_y: Half-size of patch in Y direction
            patch_size_x: Half-size of patch in X direction
            search_range_z: Search range in Z direction
            search_range_y: Search range in Y direction
            search_range_x: Search range in X direction
            correlation_threshold: Minimum correlation to accept match
        """
        # Shi-Tomasi corner detection parameters
        self.feature_params = feature_params or {
            'maxCorners': 100,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }
        
        self.min_features_ratio = min_features_ratio
        
        # 3D tracking parameters
        self.patch_size_z = patch_size_z
        self.patch_size_y = patch_size_y
        self.patch_size_x = patch_size_x
        self.search_range_z = search_range_z
        self.search_range_y = search_range_y
        self.search_range_x = search_range_x
        self.correlation_threshold = correlation_threshold
    
    def detect_features_3d(
        self,
        volume: np.ndarray,
        bbox: BoundingBox3D
    ) -> List[np.ndarray]:
        """
        Detect good features to track in 3D volume within bbox
        
        Strategy: Detect features on multiple axial slices
        
        Args:
            volume: 3D volume (depth, height, width)
            bbox: Bounding box defining ROI
            
        Returns:
            List of feature points [(N, 3)] where each point is (z, y, x)
        """
        roi = bbox.extract_from_volume(volume)
        depth, height, width = roi.shape
        
        # Select slices to detect features (every 3rd slice to avoid redundancy)
        slice_indices = range(0, depth, max(1, depth // 10))
        
        all_features = []
        
        for z_idx in slice_indices:
            # Get slice and normalize
            slice_img = roi[z_idx, :, :]
            slice_img = cv2.normalize(
                slice_img, None, 0, 255, 
                cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            # Detect corners
            corners = cv2.goodFeaturesToTrack(
                slice_img,
                mask=None,
                **self.feature_params
            )
            
            if corners is not None:
                # Convert to 3D coordinates (absolute coordinates in volume)
                corners_3d = np.zeros((len(corners), 3))
                corners_3d[:, 0] = z_idx + bbox.z_min  # Absolute Z
                corners_3d[:, 1] = corners[:, 0, 1] + bbox.y_min  # Absolute Y
                corners_3d[:, 2] = corners[:, 0, 0] + bbox.x_min  # Absolute X
                
                all_features.append(corners_3d)
        
        return all_features
    
    def track_features_3d(
        self,
        prev_volume: np.ndarray,
        curr_volume: np.ndarray,
        prev_features: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Track features in TRUE 3D using volumetric correlation
        
        Strategy:
        1. For each feature at (z, y, x) in prev_volume
        2. Extract 3D patch around it
        3. Search in 3D neighborhood in curr_volume
        4. Find best match using 3D normalized cross-correlation
        
        Args:
            prev_volume: Previous frame volume (depth, height, width)
            curr_volume: Current frame volume (depth, height, width)
            prev_features: List of feature points from previous frame
            
        Returns:
            curr_features: Tracked features in current frame
            status_list: Status of each feature (1=good, 0=lost)
            motion_vectors: Motion vectors for each feature (dz, dy, dx)
        """
        # Flatten all features
        all_prev_features = []
        for feat_array in prev_features:
            if len(feat_array) > 0:
                all_prev_features.append(feat_array)
        
        if len(all_prev_features) == 0:
            return [], [], []
        
        all_prev_features = np.vstack(all_prev_features)
        
        tracked_features = []
        tracked_status = []
        tracked_motion = []
        
        depth, height, width = prev_volume.shape
        
        for feat in all_prev_features:
            z, y, x = feat.astype(int)
            
            # Check bounds for patch extraction
            if (z - self.patch_size_z < 0 or z + self.patch_size_z >= depth or
                y - self.patch_size_y < 0 or y + self.patch_size_y >= height or
                x - self.patch_size_x < 0 or x + self.patch_size_x >= width):
                # Feature too close to boundary
                tracked_features.append(feat)
                tracked_status.append(0)  # Lost
                tracked_motion.append(np.array([0.0, 0.0, 0.0]))
                continue
            
            # Extract 3D patch from previous volume
            prev_patch = prev_volume[
                z - self.patch_size_z : z + self.patch_size_z + 1,
                y - self.patch_size_y : y + self.patch_size_y + 1,
                x - self.patch_size_x : x + self.patch_size_x + 1
            ]
            
            # Normalize patch
            prev_mean = np.mean(prev_patch)
            prev_std = np.std(prev_patch)
            if prev_std < 1e-6:
                # Homogeneous patch, skip
                tracked_features.append(feat)
                tracked_status.append(0)
                tracked_motion.append(np.array([0.0, 0.0, 0.0]))
                continue
                
            prev_patch_norm = (prev_patch - prev_mean) / prev_std
            
            # Define search region in current volume
            z_min = max(self.patch_size_z, z - self.search_range_z)
            z_max = min(depth - self.patch_size_z, z + self.search_range_z + 1)
            y_min = max(self.patch_size_y, y - self.search_range_y)
            y_max = min(height - self.patch_size_y, y + self.search_range_y + 1)
            x_min = max(self.patch_size_x, x - self.search_range_x)
            x_max = min(width - self.patch_size_x, x + self.search_range_x + 1)
            
            # Search for best match in 3D
            best_corr = -1
            best_pos = (z, y, x)
            
            for z_search in range(z_min, z_max):
                for y_search in range(y_min, y_max):
                    for x_search in range(x_min, x_max):
                        # Extract candidate patch
                        curr_patch = curr_volume[
                            z_search - self.patch_size_z : z_search + self.patch_size_z + 1,
                            y_search - self.patch_size_y : y_search + self.patch_size_y + 1,
                            x_search - self.patch_size_x : x_search + self.patch_size_x + 1
                        ]
                        
                        # Normalize
                        curr_mean = np.mean(curr_patch)
                        curr_std = np.std(curr_patch)
                        if curr_std < 1e-6:
                            continue
                            
                        curr_patch_norm = (curr_patch - curr_mean) / curr_std
                        
                        # Compute 3D normalized cross-correlation
                        corr = np.sum(prev_patch_norm * curr_patch_norm) / prev_patch_norm.size
                        
                        if corr > best_corr:
                            best_corr = corr
                            best_pos = (z_search, y_search, x_search)
            
            # Check if tracking was successful
            if best_corr > self.correlation_threshold:
                tracked_features.append(np.array(best_pos, dtype=float))
                tracked_status.append(1)  # Good
                motion = np.array(best_pos, dtype=float) - feat
                tracked_motion.append(motion)
            else:
                # Lost tracking
                tracked_features.append(feat)
                tracked_status.append(0)  # Lost
                tracked_motion.append(np.array([0.0, 0.0, 0.0]))
        
        # Return as lists for consistency
        if len(tracked_features) > 0:
            curr_features = [np.array(tracked_features)]
            status_list = [np.array(tracked_status)]
            motion_vectors = [np.array(tracked_motion)]
        else:
            curr_features = []
            status_list = []
            motion_vectors = []
        
        return curr_features, status_list, motion_vectors
    
    def estimate_global_motion(
        self,
        motion_vectors: List[np.ndarray],
        status_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate global 3D translation from tracked features
        
        Uses robust estimation (median + MAD) to find consensus motion
        
        Args:
            motion_vectors: List of motion vectors for each feature
            status_list: List of status for each feature
            
        Returns:
            global_motion: (dz, dy, dx) translation vector
            confidence: Confidence score (ratio of inliers)
        """
        if not motion_vectors or not status_list:
            return np.zeros(3), 0.0
        
        # Check if lists are empty
        if len(motion_vectors) == 0 or len(status_list) == 0:
            return np.zeros(3), 0.0
        
        # Concatenate all motion vectors
        all_motions = np.vstack(motion_vectors)
        all_status = np.hstack(status_list)
        
        # Filter by status
        valid_motions = all_motions[all_status == 1]
        
        if len(valid_motions) == 0:
            return np.zeros(3), 0.0
        
        # If very few features, just use mean
        if len(valid_motions) < 3:
            return np.mean(valid_motions, axis=0), len(valid_motions) / max(len(all_status), 1)
        
        # Remove outliers using median absolute deviation
        median_motion = np.median(valid_motions, axis=0)
        mad = np.median(np.abs(valid_motions - median_motion), axis=0)
        
        # Avoid division by zero
        mad = np.where(mad == 0, 1e-6, mad)
        
        # Threshold for inliers (3 * MAD)
        threshold = 3 * mad
        inlier_mask = np.all(
            np.abs(valid_motions - median_motion) <= threshold,
            axis=1
        )
        
        inlier_motions = valid_motions[inlier_mask]
        
        if len(inlier_motions) == 0:
            # Fallback to median if no inliers
            return median_motion, len(valid_motions) / max(len(all_status), 1)
        
        # Estimate global motion as mean of inliers
        global_motion = np.mean(inlier_motions, axis=0)
        confidence = len(inlier_motions) / max(len(valid_motions), 1)
        
        return global_motion, confidence
    
    def track_motion(
        self,
        volumes: np.ndarray,
        reference_frame_idx: int,
        reference_bbox: BoundingBox3D
    ) -> Tuple[List[BoundingBox3D], List[float]]:
        """
        Track motion across frames using 3D optical flow
        
        Unlike 2D ultrasound, we track ALL frames since we have full 3D volumes.
        We don't reject frames - instead we provide confidence scores.
        
        Args:
            volumes: All volume frames (n_frames, depth, height, width)
            reference_frame_idx: Index of reference frame
            reference_bbox: Bounding box around lesion in reference frame
            
        Returns:
            bboxes: List of bounding boxes for each frame (all valid)
            confidences: List of tracking confidence values (0-1)
        """
        n_frames = volumes.shape[0]
        
        # Initialize output - ALL frames will have valid bboxes
        tracked_bboxes = [None] * n_frames
        confidences = [0.0] * n_frames
        
        # Reference frame
        tracked_bboxes[reference_frame_idx] = reference_bbox
        confidences[reference_frame_idx] = 1.0
        
        # Detect features in reference frame
        print(f"\nDetecting features in reference frame...")
        ref_features = self.detect_features_3d(
            volumes[reference_frame_idx],
            reference_bbox
        )
        
        total_features = sum(len(f) for f in ref_features)
        print(f"  Detected {total_features} features across {len(ref_features)} slices")
        
        # Track forward from reference frame
        print(f"\nTracking forward...")
        current_bbox = reference_bbox
        current_features = ref_features
        
        for frame_idx in range(reference_frame_idx + 1, n_frames):
            print(f"  Frame {frame_idx}/{n_frames-1}...", end='\r')
            
            # Track features using TRUE 3D
            next_features, status, motion_vectors = self.track_features_3d(
                volumes[frame_idx - 1],
                volumes[frame_idx],
                current_features
            )
            
            # Estimate global motion
            global_motion, confidence = self.estimate_global_motion(
                motion_vectors,
                status
            )
            
            # ALWAYS apply translation (no frame rejection)
            dz, dy, dx = global_motion.astype(int)
            current_bbox = current_bbox.translate(dz, dy, dx)
            
            # Update features for next iteration
            total_features = sum(len(f) for f in next_features) if next_features else 0
            
            if confidence < self.min_features_ratio or total_features < 10:
                # Re-detect features in current bbox to avoid drift
                print(f"\n  Frame {frame_idx}: Low confidence ({confidence:.2f}) or few features ({total_features}), re-detecting...")
                current_features = self.detect_features_3d(
                    volumes[frame_idx],
                    current_bbox
                )
            else:
                current_features = next_features
            
            tracked_bboxes[frame_idx] = current_bbox
            confidences[frame_idx] = confidence
        
        print()  # New line after progress
        
        # Track backward from reference frame
        print(f"\nTracking backward...")
        current_bbox = reference_bbox
        current_features = ref_features
        
        for frame_idx in range(reference_frame_idx - 1, -1, -1):
            print(f"  Frame {frame_idx}...", end='\r')
            
            # Track features using TRUE 3D
            next_features, status, motion_vectors = self.track_features_3d(
                volumes[frame_idx + 1],
                volumes[frame_idx],
                current_features
            )
            
            # Estimate global motion
            global_motion, confidence = self.estimate_global_motion(
                motion_vectors,
                status
            )
            
            # ALWAYS apply translation
            dz, dy, dx = global_motion.astype(int)
            current_bbox = current_bbox.translate(dz, dy, dx)
            
            # Update features
            total_features = sum(len(f) for f in next_features) if next_features else 0
            
            if confidence < self.min_features_ratio or total_features < 10:
                print(f"\n  Frame {frame_idx}: Low confidence ({confidence:.2f}), re-detecting...")
                current_features = self.detect_features_3d(
                    volumes[frame_idx],
                    current_bbox
                )
            else:
                current_features = next_features
            
            tracked_bboxes[frame_idx] = current_bbox
            confidences[frame_idx] = confidence
        
        print()  # New line
        print("\nTracking complete!")
        
        return tracked_bboxes, confidences


def visualize_optical_flow_tracking(
    volumes: np.ndarray,
    tracked_bboxes: List[BoundingBox3D],
    confidences: List[float],
    output_path: str = 'optical_flow_tracking.png'
):
    """
    Visualize optical flow tracking results
    
    Args:
        volumes: All volume frames
        tracked_bboxes: List of tracked bounding boxes (all valid)
        confidences: List of confidence values
        output_path: Path to save visualization
    """
    n_frames = len(tracked_bboxes)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot confidence over time
    frames = np.arange(n_frames)
    axes[0].plot(frames, confidences, 'b-o', linewidth=2, markersize=6)
    axes[0].axhline(0.3, color='r', linestyle='--', alpha=0.5, 
                   label='Low confidence threshold')
    axes[0].set_xlabel('Frame Number', fontsize=12)
    axes[0].set_ylabel('Tracking Confidence', fontsize=12)
    axes[0].set_title('3D Optical Flow Tracking Confidence', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Plot bbox center movement (all frames valid)
    centers = np.array([bbox.center for bbox in tracked_bboxes])
    
    axes[1].plot(frames, centers[:, 0], 'r-', label='Z position', alpha=0.7, linewidth=2)
    axes[1].plot(frames, centers[:, 1], 'g-', label='Y position', alpha=0.7, linewidth=2)
    axes[1].plot(frames, centers[:, 2], 'b-', label='X position', alpha=0.7, linewidth=2)
    axes[1].set_xlabel('Frame Number', fontsize=12)
    axes[1].set_ylabel('Bbox Center Position (pixels)', fontsize=12)
    axes[1].set_title('3D Bounding Box Movement', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved tracking visualization to {output_path}")