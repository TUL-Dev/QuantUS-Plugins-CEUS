from typing import Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def resample_to_spacing(image_arr: np.ndarray, original_spacing: Tuple[float, float, float], 
                        new_spacing: Tuple[float, float, float], interp='linear') -> sitk.Image:
    """Resample to isotropic/anisotropic spacing by recomputing size.
    Maintains image FOV; origin/direction preserved.
    """
    assert image_arr.ndim == 4 or image_arr.ndim == 3, "Image array must be 4D (x, y, z, t) or 3D (x, y, z)."
    
    affine = np.eye(4)
    reversed_dims = list(reversed(original_spacing)) # Ensure spacing is in (z, y, x) order for SimpleITK
    for i in range(3):
        affine[i, i] = reversed_dims[i]
    origin = affine[:3, -1].tolist()
    direction_matrix = affine[:3, :3]
    spacing = np.linalg.norm(direction_matrix, axis=0).tolist()
    direction = (direction_matrix / spacing).flatten().tolist()

    new_spacing = list(reversed(new_spacing))  # Ensure new_spacing is in (z, y, x) order for SimpleITK

    if image_arr.ndim == 3:
        image_arr = np.expand_dims(image_arr, axis=-1)

    if interp == 'linear':
        interpolator = sitk.sitkLinear
    elif interp == 'nearest':
        interpolator = sitk.sitkNearestNeighbor
    elif interp == 'cubic':
        interpolator = sitk.sitkBSpline
    else:
        raise ValueError("Interpolation method must be one of 'linear', 'nearest', or 'cubic'.")
    
    resampled_frames = []
    for i in tqdm(range(image_arr.shape[3]), desc="Resampling frames"):
        frame = sitk.GetImageFromArray(image_arr[:, :, :, i])
        frame.SetSpacing(spacing)
        frame.SetOrigin(origin)
        frame.SetDirection(direction)

        original_size = np.array(list(frame.GetSize()), dtype=int)
        new_size = np.round(original_size * (np.array(list(frame.GetSpacing())) / np.array(new_spacing))).astype(int)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(tuple(new_spacing))
        resampler.SetSize([int(x) for x in new_size])
        resampler.SetOutputOrigin(frame.GetOrigin())
        resampler.SetOutputDirection(frame.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(interpolator)

        out = resampler.Execute(frame)
        out = sitk.Cast(out, frame.GetPixelID())
        resampled_frames.append(sitk.GetArrayFromImage(out))

    out = np.stack(resampled_frames, axis=-1).squeeze()
    return out
