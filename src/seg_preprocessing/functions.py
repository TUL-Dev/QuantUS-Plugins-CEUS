import numpy as np
import json
from pathlib import Path
from scipy.ndimage import shift

from ..seg_preprocessing.decorators import required_kwargs
from ..data_objs.image import UltrasoundImage
from ..data_objs.seg import CeusSeg
from ..image_preprocessing.transforms import resample_to_spacing_2d, resample_to_spacing_3d
from ..seg_preprocessing.motion_compensation_3d import MotionCompensation3D, BoundingBox3D, OpticalFlowMotionCompensation3D

@required_kwargs('target_vox_size', 'interp')
def resample(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
    """
    Resample the image data to a new spacing.

    Kwargs:
        target_vox_size: tuple of (z, y, x) spacing in mm to resample the image to.
        interp: interpolation method, one of 'nearest', 'linear', 'cubic'.
    """
    target_vox_size = kwargs['target_vox_size']
    interp = kwargs['interp']
    if 'original_spacing' in image_data.extras_dict.keys():
        pixdim = image_data.extras_dict['original_spacing']
    else:
        pixdim = image_data.pixdim

    if seg_data.seg_mask.ndim == 3:
        seg_data.seg_mask = resample_to_spacing_3d(seg_data.seg_mask, pixdim, target_vox_size, interp=interp)
    elif seg_data.seg_mask.ndim == 2:
        seg_data.seg_mask = resample_to_spacing_2d(seg_data.seg_mask, pixdim, target_vox_size, interp=interp)
    else:
        raise ValueError("Segmentation mask must be either 2D or 3D.")

    return seg_data

@required_kwargs('motion_json_path','search_margin_ratio','padding')
def apply_motion_compensation(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
    """
    Apply motion compensation to segmentation mask based on translational motion vectors.

    The motion vectors are read from a JSON file where each key is a frame number and
    each value is a list [z, y, x] representing the translational movement in pixels
    for that frame.

    Kwargs:
        motion_json_path: Path to JSON file containing motion vectors.
                         Format: {"0": [z, y, x], "1": [z, y, x], ...}
        order: (optional) Interpolation order for shifting. Default is 0 (nearest neighbor).
               0 = nearest, 1 = linear, 3 = cubic
        cval: (optional) Value to fill borders. Default is 0.
        prefilter: (optional) Whether to apply spline filter before interpolation. Default is True.

    Returns:
        CeusSeg: Segmentation data with motion-compensated mask

    Note:
        - The function expects seg_mask to be 4D (x, y, z, frames) or 3D (x, y, frames)
        - Motion vectors should match the number of frames in the segmentation
        - Positive values shift in positive direction, negative values shift in negative direction
    """
    motion_json_path = kwargs['motion_json_path']
    order = kwargs.get('order', 0)  # Default to nearest neighbor for segmentation masks
    cval = kwargs.get('cval', 0)
    prefilter = kwargs.get('prefilter', True)

    # Load motion vectors from JSON file
    motion_json_path = Path(motion_json_path)
    if not motion_json_path.exists():
        raise FileNotFoundError(f"Motion JSON file not found: {motion_json_path}")

    with open(motion_json_path, 'r') as f:
        motion_vectors = json.load(f)

    # Convert string keys to integers and sort
    motion_dict = {int(k): v for k, v in motion_vectors.items()}

    seg_mask = seg_data.seg_mask

    # Check if segmentation mask has time dimension
    if seg_mask.ndim not in [3, 4]:
        raise ValueError(f"Segmentation mask must be 3D (x, y, frames) or 4D (x, y, z, frames), got {seg_mask.ndim}D")

    # Get number of frames (last dimension)
    num_frames = seg_mask.shape[-1]

    # Verify motion vectors match number of frames
    if len(motion_dict) != num_frames:
        raise ValueError(f"Number of motion vectors ({len(motion_dict)}) does not match number of frames ({num_frames})")

    # Apply motion compensation frame by frame
    compensated_mask = np.zeros_like(seg_mask)

    for frame_idx in range(num_frames):
        if frame_idx not in motion_dict:
            raise KeyError(f"Motion vector for frame {frame_idx} not found in JSON file")

        motion_vector = motion_dict[frame_idx]

        # Get the current frame (extract along last dimension)
        if seg_mask.ndim == 4:  # 4D: (x, y, z, frames)
            current_frame = seg_mask[..., frame_idx]
        else:  # 3D: (x, y, frames)
            current_frame = seg_mask[..., frame_idx]

        # Prepare shift vector based on dimensionality
        if seg_mask.ndim == 4:  # 4D: (x, y, z, frames)
            if len(motion_vector) != 3:
                raise ValueError(f"Motion vector for frame {frame_idx} must have 3 components [z, y, x], got {len(motion_vector)}")
            # Motion vector is [z, y, x], but numpy array is indexed as [x, y, z]
            # So we need to reverse and negate to compensate for motion
            shift_vector = [motion_vector[2], motion_vector[1], motion_vector[0]]  # [x, y, z]
        elif seg_mask.ndim == 3:  # 3D: (x, y, frames) TODO need to improve this
            if len(motion_vector) < 2:
                raise ValueError(f"Motion vector for frame {frame_idx} must have at least 2 components [y, x] or [z, y, x], got {len(motion_vector)}")
            # Use y and x components - reverse and negate to compensate for motion
            if len(motion_vector) == 3:
                shift_vector = [motion_vector[2], motion_vector[1]]  # [x, y] from [z, y, x]
            else:
                shift_vector = [motion_vector[1], motion_vector[0]]  # [x, y] from [y, x]

        # Apply shift to compensate for motion
        shifted_frame = shift(
            current_frame,
            shift=shift_vector,
            order=order,
            cval=cval,
            prefilter=prefilter
        )

        # Store the compensated frame
        if seg_mask.ndim == 4:
            compensated_mask[..., frame_idx] = shifted_frame
        else:
            compensated_mask[..., frame_idx] = shifted_frame

    # Update segmentation mask
    seg_data.mc_seg_mask = compensated_mask
    seg_data.use_mc = True

    return seg_data

# ==================== Integration with QuantUS Pipeline ====================
@required_kwargs('bmode_image_data','search_margin_ratio','padding')
def motion_compensation_3d(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
    """
    Apply 3D motion compensation using ILSA tracking.
    
    Creates seg_data.mc_seg_mask with motion compensation applied.
    
    Kwargs:
        bmode_image_data (UltrasoundImage): B-mode data for motion tracking [REQUIRED]
        reference_frame (int): Reference frame index (default: 0)
        search_margin_ratio (float): Search margin ratio (default: 0.5/30)
        padding (int): Padding around bounding box (default: 5)
        shift_order (int): Interpolation order for shifting (default: 0 for nearest neighbor)
    
    Returns:
        CeusSeg: Segmentation with mc_seg_mask created
    """
    # Extract kwargs
    bmode_image_data = kwargs['bmode_image_data']
    reference_frame = kwargs.get('reference_frame', 0)
    search_margin_ratio = kwargs.get('search_margin_ratio', 0.5 / 25)
    padding = kwargs.get('padding', 5)
    shift_order = kwargs.get('shift_order', 0)  # 0=nearest neighbor for masks
    
    # Validate inputs
    if not isinstance(bmode_image_data, UltrasoundImage):
        raise TypeError("bmode_image_data must be an UltrasoundImage object")
    
    bmode_shape = bmode_image_data.pixel_data.shape
    if bmode_image_data.pixel_data.ndim != 4:
        raise ValueError(f"B-mode data must be 4D (T, Z, Y, X), got shape {bmode_shape}")
    
    reference_mask = seg_data.seg_mask[:,:,:,0]
    # seg_mask should be (Z, Y, X) - single frame
    seg_mask_shape = reference_mask.shape
    if reference_mask.ndim != 3:
        raise ValueError(f"Segmentation mask must be 3D (Z, Y, X), got shape {seg_mask_shape}")
    
    print("\n" + "="*60)
    print("3D Motion Compensation with ILSA Tracking")
    print("="*60)
    
    # Step 1: Extract bounding box from segmentation
    print("\nStep 1: Extracting bounding box from segmentation...")
    try:
        reference_bbox = BoundingBox3D.from_mask(reference_mask, padding=padding)
        print(f"  Bounding box: Z=[{reference_bbox.z_min}, {reference_bbox.z_max}], "
              f"Y=[{reference_bbox.y_min}, {reference_bbox.y_max}], "
              f"X=[{reference_bbox.x_min}, {reference_bbox.x_max}]")
        print(f"  Center: {reference_bbox.center}")
    except ValueError as e:
        print(f"Error: {e}")
        return seg_data
    
    # Step 2: Track motion using ILSA
    print("\nStep 2: Tracking motion using ILSA...")
    print(f"  Reference frame: {reference_frame}")
    print(f"  Search margin ratio: {search_margin_ratio}")
    
    mc = MotionCompensation3D(search_margin_ratio=search_margin_ratio)
    
    # Track motion - volumes are (T, Z, Y, X)
    tracked_bboxes, correlations = mc.track_motion_ilsa_3d(
        bmode_image_data.pixel_data,
        reference_frame,
        reference_bbox
    )
    
    # Step 3: Apply motion compensation to create mc_seg_mask
    print("\nStep 3: Applying motion compensation to segmentation...")
    n_frames = bmode_shape[-1]
    
    # Create mc_seg_mask with shape (Z, Y, X, T)
    mc_seg_mask = np.zeros((*seg_mask_shape, n_frames), dtype=seg_data.seg_mask.dtype)
    
    # Get reference center
    ref_center = reference_bbox.center
    
    for frame_idx in range(n_frames):
        bbox = tracked_bboxes[frame_idx]
        
        # Calculate shift from reference
        curr_center = bbox.center
        shift_z = curr_center[0] - ref_center[0]
        shift_y = curr_center[1] - ref_center[1]
        shift_x = curr_center[2] - ref_center[2]
        
        # Apply shift to segmentation (shift in opposite direction to compensate)
        # Note: seg_mask is (Z, Y, X), so shift vector is [z, y, x]
        shifted_mask = shift(
            seg_data.seg_mask[...,frame_idx],
            shift=[shift_z, shift_y, shift_x],  # Negative to compensate
            order=shift_order,
            cval=0,
            prefilter=True if shift_order > 0 else False
        )
        
        # Store in mc_seg_mask - last dimension is time
        mc_seg_mask[..., frame_idx] = shifted_mask
        
        if frame_idx % 10 == 0 or frame_idx == n_frames - 1:
            print(f"  Frame {frame_idx}: shift=({shift_z:.1f}, {shift_y:.1f}, {shift_x:.1f}), "
                  f"corr={correlations[frame_idx]:.3f}")
    
    # Store results
    seg_data.mc_seg_mask = mc_seg_mask
    seg_data.use_mc = True
    
    # Store motion info in extras_dict
    image_data.extras_dict['motion_compensation'] = {
        'applied': True,
        'reference_frame': reference_frame,
        'mean_correlation': float(np.mean(correlations)),
        'min_correlation': float(np.min(correlations)),
        'bboxes': [
            {
                'z_min': bbox.z_min, 'z_max': bbox.z_max,
                'y_min': bbox.y_min, 'y_max': bbox.y_max,
                'x_min': bbox.x_min, 'x_max': bbox.x_max,
                'center': bbox.center
            } for bbox in tracked_bboxes
        ],
        'correlations': [float(c) for c in correlations]
    }
    
    print("\n" + "="*60)
    print("Motion Compensation Complete!")
    print(f"  mc_seg_mask shape: {mc_seg_mask.shape}")
    print(f"  Mean correlation: {np.mean(correlations):.3f}")
    print("="*60 + "\n")
    
    return seg_data

# Then add a new plugin function
@required_kwargs('bmode_image_data')
def motion_compensation_3d_optical_flow(image_data: UltrasoundImage, seg_data: CeusSeg, **kwargs) -> CeusSeg:
    """
    Apply 3D motion compensation using optical flow feature tracking.
    
    Kwargs:
        bmode_image_data (UltrasoundImage): B-mode data for motion tracking [REQUIRED]
        reference_frame (int): Reference frame index (default: 0)
        padding (int): Padding around bounding box (default: 5)
        shift_order (int): Interpolation order for shifting (default: 0)
        # Optical flow specific parameters:
        max_corners (int): Maximum corners to detect (default: 100)
        quality_level (float): Corner quality (default: 0.3)
        patch_size_z (int): Z patch size (default: 5)
        patch_size_y (int): Y patch size (default: 7)
        patch_size_x (int): X patch size (default: 7)
        search_range_z (int): Z search range (default: 3)
        search_range_y (int): Y search range (default: 7)
        search_range_x (int): X search range (default: 7)
    
    Returns:
        CeusSeg: Segmentation with mc_seg_mask created
    """
    # Extract kwargs
    bmode_image_data = kwargs['bmode_image_data']
    reference_frame = kwargs.get('reference_frame', 0)
    padding = kwargs.get('padding', 5)
    shift_order = kwargs.get('shift_order', 0)
    
    # Optical flow parameters
    feature_params = {
        'maxCorners': kwargs.get('max_corners', 100),
        'qualityLevel': kwargs.get('quality_level', 0.3),
        'minDistance': kwargs.get('min_distance', 7),
        'blockSize': kwargs.get('block_size', 7)
    }
    
    patch_size_z = kwargs.get('patch_size_z', 5)
    patch_size_y = kwargs.get('patch_size_y', 7)
    patch_size_x = kwargs.get('patch_size_x', 7)
    search_range_z = kwargs.get('search_range_z', 3)
    search_range_y = kwargs.get('search_range_y', 7)
    search_range_x = kwargs.get('search_range_x', 7)
    
    # Validate inputs
    if not isinstance(bmode_image_data, UltrasoundImage):
        raise TypeError("bmode_image_data must be an UltrasoundImage object")
    
    bmode_shape = bmode_image_data.pixel_data.shape
    if bmode_image_data.pixel_data.ndim != 4:
        raise ValueError(f"B-mode data must be 4D (Z, Y, X, T), got shape {bmode_shape}")

    reference_mask = seg_data.seg_mask[:,:,:,0]
    seg_mask_shape = reference_mask.shape
    if reference_mask.ndim != 3:
        raise ValueError(f"Segmentation mask must be 3D (Z, Y, X), got shape {seg_mask_shape}")
    
    print("\n" + "="*60)
    print("3D Motion Compensation with Optical Flow")
    print("="*60)
    
    # Step 1: Extract bounding box
    print("\nStep 1: Extracting bounding box from segmentation...")
    try:
        reference_bbox = BoundingBox3D.from_mask(reference_mask, padding=padding)
        print(f"  Bounding box: Z=[{reference_bbox.z_min}, {reference_bbox.z_max}], "
              f"Y=[{reference_bbox.y_min}, {reference_bbox.y_max}], "
              f"X=[{reference_bbox.x_min}, {reference_bbox.x_max}]")
    except ValueError as e:
        print(f"Error: {e}")
        return seg_data
    
    # Step 2: Track motion using optical flow
    print("\nStep 2: Tracking motion using 3D optical flow...")
    
    mc = OpticalFlowMotionCompensation3D(
        feature_params=feature_params,
        patch_size_z=patch_size_z,
        patch_size_y=patch_size_y,
        patch_size_x=patch_size_x,
        search_range_z=search_range_z,
        search_range_y=search_range_y,
        search_range_x=search_range_x
    )
    
    # Transpose volumes from (Z, Y, X, T) to (T, Z, Y, X) for tracking
    volumes_transposed = np.transpose(bmode_image_data.pixel_data, (3, 0, 1, 2))
    
    tracked_bboxes, confidences = mc.track_motion(
        volumes_transposed,
        reference_frame,
        reference_bbox
    )
    
    # Step 3: Apply motion compensation
    print("\nStep 3: Applying motion compensation to segmentation...")
    n_frames = bmode_shape[-1]
    mc_seg_mask = np.zeros((*seg_mask_shape, n_frames), dtype=reference_mask.dtype)
    
    ref_center = reference_bbox.center
    
    for frame_idx in range(n_frames):
        bbox = tracked_bboxes[frame_idx]
        curr_center = bbox.center
        
        shift_z = curr_center[0] - ref_center[0]
        shift_y = curr_center[1] - ref_center[1]
        shift_x = curr_center[2] - ref_center[2]
        
        shifted_mask = shift(
            seg_data.seg_mask[..., frame_idx],
            shift=[-shift_z, -shift_y, -shift_x],
            order=shift_order,
            cval=0,
            prefilter=True if shift_order > 0 else False
        )
        
        mc_seg_mask[..., frame_idx] = shifted_mask
        
        if frame_idx % 10 == 0 or frame_idx == n_frames - 1:
            print(f"  Frame {frame_idx}: shift=({shift_z:.1f}, {shift_y:.1f}, {shift_x:.1f}), "
                  f"confidence={confidences[frame_idx]:.3f}")
    
    # Store results
    seg_data.mc_seg_mask = mc_seg_mask
    seg_data.use_mc = True
    
    image_data.extras_dict['motion_compensation'] = {
        'applied': True,
        'method': 'optical_flow',
        'reference_frame': reference_frame,
        'mean_confidence': float(np.mean(confidences)),
        'min_confidence': float(np.min(confidences)),
        'confidences': [float(c) for c in confidences]
    }
    
    print("\n" + "="*60)
    print("Motion Compensation Complete!")
    print(f"  mc_seg_mask shape: {mc_seg_mask.shape}")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print("="*60 + "\n")
    
    return seg_data