from skimage import exposure, filters
from skimage.restoration import denoise_wavelet
from scipy.ndimage import median_filter

import numpy as np
import cv2
import sys

from typing import Dict, Any, List, Tuple
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QSlider, QGroupBox, QScrollArea, QPushButton,
    QDoubleSpinBox, QSpinBox, QFrame, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import nibabel as nib
import threading

def enhance_image(volume, method='clahe', **kwargs):
    """
    Enhance image quality using various methods
    
    Args:
        volume: 3D volume (Z, Y, X)
        method: Enhancement method ('clahe', 'gamma', 'log', 'sigmoid', 'adaptive_hist')
    """
    
    enhanced = np.zeros_like(volume)
    if method == 'gamma':
        # Gamma correction
        gamma = kwargs.get('gamma', 0.7)
        enhanced = exposure.adjust_gamma(volume, gamma)
    elif method == 'adaptive_hist':
        clip_limit = kwargs.get('clip_limit', 0.05)
        enhanced = exposure.equalize_adapthist(volume, clip_limit=clip_limit)

    for z in range(volume.shape[2]):
        slice_2d = volume[:, :, z]
            
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=kwargs.get('clip_limit',3.0), tileGridSize=(8,8))
            enhanced[:,:,z] = clahe.apply(slice_2d)
            
        elif method == 'sigmoid':
            # Sigmoid transformation
            cutoff = kwargs.get('cutoff', 0.5)
            gain = kwargs.get('gain', 10)
            enhanced[:,:,z] = exposure.adjust_sigmoid(slice_2d, cutoff=cutoff, gain=gain)      
    return enhanced

def imsharpen(image, radius=1.5, amount=0.5):
    """
    Python equivalent of MATLAB's imsharpen function
    
    Args:
        image: Input image (2D or 3D)
        radius: Gaussian blur radius (equivalent to MATLAB 'Radius')
        amount: Sharpening strength (equivalent to MATLAB 'Amount')
    
    Returns:
        Sharpened image
    """
    if image.ndim == 3:
        # Process 3D volume slice by slice
        sharpened = np.zeros_like(image, dtype=np.float64)
        for z in range(image.shape[0]):
            # Convert to float for processing
            slice_float = image[z].astype(np.float64)
            
            # Create Gaussian blurred version
            blurred = filters.gaussian(slice_float, sigma=radius)
            
            # Apply unsharp mask: original + amount * (original - blurred)
            sharpened[z] = slice_float + amount * (slice_float - blurred)
            
    else:
        # Process 2D image
        slice_float = image.astype(np.float64)
        blurred = filters.gaussian(slice_float, sigma=radius)
        sharpened = slice_float + amount * (slice_float - blurred)
    
    # Clip to valid range and convert back to original dtype
    sharpened = np.clip(sharpened, 0, np.max(image))
    
    return sharpened.astype(image.dtype)

def denoise_ceus_wavelet(volume_3d, wavelet='db1', sigma_scale=0.8):
    """
    Gentler wavelet denoising
    
    Args:
        sigma_scale: Scale factor for noise estimate (0.3-0.7 for gentle, 1.0 for normal)
    """
    denoised = np.zeros_like(volume_3d, dtype=np.float32)
    
    for z in range(volume_3d.shape[2]):
        slice_2d = volume_3d[:, :, z].astype(np.float32)
        
        # Normalize to [0, 1]
        slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
        
        # Estimate sigma and scale it down
        from skimage.restoration import estimate_sigma
        sigma_est = estimate_sigma(slice_norm, average_sigmas=True)
        
        # Apply wavelet denoising with reduced sigma
        denoised_slice = denoise_wavelet(
            slice_norm,
            method='BayesShrink',
            mode='soft',
            wavelet=wavelet,
            rescale_sigma=True,
            sigma=sigma_est * sigma_scale  # KEY: Reduce denoising strength
        )
        
        # Scale back
        denoised[:, :, z] = denoised_slice * (slice_2d.max() - slice_2d.min()) + slice_2d.min()
    
    return denoised



def anisotropic_diffusion_3d(volume_3d, niter=2, kappa=30, gamma=0.15, option=1):
    """
    3D Anisotropic Diffusion for edge-preserving smoothing
    
    Args:
        volume_3d: Input 3D volume
        niter: Number of iterations (10-20 for gentle, 20-30 for moderate)
        kappa: Edge threshold (30-40 for ultrasound)
        gamma: Diffusion rate (0.1-0.2 safe for 3D, must be ≤0.25)
        option: 1=sharp edges, 2=wide regions (better for ultrasound)
    
    Returns:
        Smoothed volume with preserved edges
    """
    
    img = volume_3d.astype(np.float32)
    
    for _ in range(niter):
        # Calculate gradients in 6 directions
        nabla_n = np.roll(img, 1, axis=0) - img
        nabla_s = np.roll(img, -1, axis=0) - img
        nabla_e = np.roll(img, 1, axis=1) - img
        nabla_w = np.roll(img, -1, axis=1) - img
        nabla_u = np.roll(img, 1, axis=2) - img
        nabla_d = np.roll(img, -1, axis=2) - img
        
        # Calculate conduction coefficients
        if option == 1:
            c_n = np.exp(-(nabla_n/kappa)**2)
            c_s = np.exp(-(nabla_s/kappa)**2)
            c_e = np.exp(-(nabla_e/kappa)**2)
            c_w = np.exp(-(nabla_w/kappa)**2)
            c_u = np.exp(-(nabla_u/kappa)**2)
            c_d = np.exp(-(nabla_d/kappa)**2)
        else:  # option == 2
            c_n = 1.0 / (1.0 + (nabla_n/kappa)**2)
            c_s = 1.0 / (1.0 + (nabla_s/kappa)**2)
            c_e = 1.0 / (1.0 + (nabla_e/kappa)**2)
            c_w = 1.0 / (1.0 + (nabla_w/kappa)**2)
            c_u = 1.0 / (1.0 + (nabla_u/kappa)**2)
            c_d = 1.0 / (1.0 + (nabla_d/kappa)**2)
        
        # Update image
        img += gamma * (c_n*nabla_n + c_s*nabla_s + c_e*nabla_e + 
                       c_w*nabla_w + c_u*nabla_u + c_d*nabla_d)
    
    return img


def speckle_reduction_bias_field(
    volume_3d: np.ndarray,
    kernel_size: int = 7,
    weight: float = 0.3,
    adjust_dynamic_range: bool = False
) -> np.ndarray:
    """
    Reduce speckle visibility using bias field approach from the paper
    
    This method adds a local brightness bias to reduce speckle contrast,
    making it less visible to human observers without destroying the pattern.
    
    Args:
        volume_3d: Input 3D volume (Y, X, Z) or (Z, Y, X)
        kernel_size: Size of median filter kernel (paper uses 9)
                    Larger = smoother bias field, more speckle reduction
                    Smaller = preserves more local detail
        weight: Blending weight for bias field (0 to 1)
               - 0: No effect (original image)
               - 0.5: Balanced (paper's default)
               - 1.0: Maximum speckle reduction
        adjust_dynamic_range: If True, rescale output to full 8-bit range
    
    Returns:
        Denoised 3D volume with reduced speckle visibility
    """
    
    denoised = np.zeros_like(volume_3d, dtype=np.float32)
    
    for z in range(volume_3d.shape[2]):
        slice_2d = volume_3d[:, :, z].astype(np.float32)
        
        # Step 1: Estimate bias field using median filter
        # This creates a smoothed version that captures local brightness
        bias_field = median_filter(slice_2d, size=kernel_size)
        
        # Step 2: Add bias field to original with user-controlled weight
        # This increases local brightness, reducing speckle contrast
        enhanced = slice_2d + weight * bias_field
        
        # Step 3: Adjust dynamic range if requested
        if adjust_dynamic_range:
            # Normalize to [0, 255] range
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
            enhanced = enhanced * 255
        
        denoised[:, :, z] = enhanced
    
    # Convert back to original dtype if it was uint8
    if volume_3d.dtype == np.uint8:
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    
    return denoised