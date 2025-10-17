import numpy as np
import json
from pathlib import Path
from scipy.ndimage import shift

from .decorators import required_kwargs
from ..data_objs.image import UltrasoundImage
from ..data_objs.seg import CeusSeg
from ..image_preprocessing.transforms import resample_to_spacing_2d, resample_to_spacing_3d

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

@required_kwargs('motion_json_path')
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

    return seg_data