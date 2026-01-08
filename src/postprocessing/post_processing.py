from skimage import exposure, filters
from skimage.restoration import denoise_wavelet
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



class FilterListWidget(QWidget):
    """
    Widget to manage a list of filters with add/remove buttons
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filters = []
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Filter list
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(150)
        layout.addWidget(QLabel("Active Filters:"))
        layout.addWidget(self.list_widget)
        
        # Add filter section
        add_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "CLAHE",
            "Gamma Correction",
            "Bilateral Filter",
            "Gaussian Blur",
            "Median Filter",
            "Shock Filter",
            "Ridge Enhancement",
            "Speckle Reduction",
            "Unsharp Mask",
            "Directional Filter"
        ])
        add_layout.addWidget(self.filter_combo)
        
        add_btn = QPushButton("+")
        add_btn.setMaximumWidth(40)
        add_btn.clicked.connect(self.add_filter)
        add_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("-")
        remove_btn.setMaximumWidth(40)
        remove_btn.clicked.connect(self.remove_filter)
        add_layout.addWidget(remove_btn)
        
        layout.addLayout(add_layout)
        
        self.setLayout(layout)
    
    def add_filter(self):
        """Add selected filter to the list"""
        filter_name = self.filter_combo.currentText()
        
        # Get default parameters
        params = self.get_default_params(filter_name)
        
        self.filters.append(params)
        self.list_widget.addItem(filter_name)
        
        # Notify parent
        if hasattr(self.parent(), 'on_filters_changed'):
            self.parent().on_filters_changed()
    
    def remove_filter(self):
        """Remove selected filter from the list"""
        current_row = self.list_widget.currentRow()
        if current_row >= 0:
            self.list_widget.takeItem(current_row)
            self.filters.pop(current_row)
            
            # Notify parent
            if hasattr(self.parent(), 'on_filters_changed'):
                self.parent().on_filters_changed()
    
    def get_default_params(self, filter_name: str) -> Dict:
        """Get default parameters for a filter"""
        defaults = {
            "CLAHE": {'name': 'CLAHE', 'clip_limit': 2.0, 'tile_size': 8},
            "Gamma Correction": {'name': 'Gamma', 'gamma': 1.2},
            "Bilateral Filter": {'name': 'Bilateral', 'd': 9, 'sigma_color': 75, 'sigma_space': 75},
            "Gaussian Blur": {'name': 'Gaussian', 'sigma': 1.5},
            "Median Filter": {'name': 'Median', 'ksize': 5},
            "Shock Filter": {'name': 'Shock', 'iterations': 2},
            "Ridge Enhancement": {'name': 'Ridge', 'threshold': 0.3, 'boost': 1.5},
            "Speckle Reduction": {'name': 'Speckle', 'h': 10, 'template': 7, 'search': 21},
            "Unsharp Mask": {'name': 'Unsharp', 'sigma': 1.0, 'amount': 0.5},
            "Directional Filter": {'name': 'Directional', 'n_orientations': 8, 'sigma': 2.0}
        }
        return defaults.get(filter_name, {})
    
    def clear_filters(self):
        """Clear all filters"""
        self.filters = []
        self.list_widget.clear()


class PyramidLevelPanel(QWidget):
    """
    Compact panel for one pyramid level
    """
    
    def __init__(self, level_idx: int, level_volume: np.ndarray, parent=None):
        super().__init__(parent)
        self.level_idx = level_idx
        self.original_volume = level_volume.copy()
        self.current_volume = level_volume.copy()
        self.z_slice_idx = level_volume.shape[2] // 2
        
        self.setup_ui()
        self.update_image()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Title
        title = QLabel(f"Level {self.level_idx}")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Image display (smaller)
        self.image_label = QLabel()
        self.image_label.setFixedSize(250, 250)
        self.image_label.setScaledContents(True)
        self.image_label.setFrameStyle(QFrame.Shape.Box)
        layout.addWidget(self.image_label)
        
        # Z-slice slider (compact)
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(self.original_volume.shape[2] - 1)
        self.z_slider.setValue(self.z_slice_idx)
        self.z_slider.valueChanged.connect(self.on_z_changed)
        z_layout.addWidget(self.z_slider)
        self.z_label = QLabel(f"{self.z_slice_idx}")
        self.z_label.setFixedWidth(30)
        z_layout.addWidget(self.z_label)
        layout.addLayout(z_layout)
        
        # Filter list widget
        self.filter_list = FilterListWidget(self)
        layout.addWidget(self.filter_list)
        
        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def on_z_changed(self, value):
        self.z_slice_idx = value
        self.z_label.setText(str(value))
        self.update_image()
    
    def on_filters_changed(self):
        """Called when filters are added/removed"""
        self.apply_all_filters()
    
    def apply_all_filters(self):
        """Apply all filters in sequence"""
        self.current_volume = self.original_volume.copy()
        
        for filter_params in self.filter_list.filters:
            self.current_volume = self.apply_single_filter(self.current_volume, filter_params)
        
        self.update_image()
        
        # Notify main window
        if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), 'update_final_blend'):
            self.parent().parent().update_final_blend()
    
    def apply_single_filter(self, volume: np.ndarray, params: Dict) -> np.ndarray:
        """Apply a single filter to the volume"""
        result = np.zeros_like(volume, dtype=np.float32)
        
        for z in range(volume.shape[2]):
            slice_2d = volume[:, :, z].astype(np.float32)
            
            if params['name'] == 'CLAHE':
                slice_uint8 = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                clahe = cv2.createCLAHE(
                    clipLimit=params['clip_limit'],
                    tileGridSize=(params['tile_size'], params['tile_size'])
                )
                slice_2d = clahe.apply(slice_uint8).astype(np.float32)
            
            elif params['name'] == 'Gamma':
                slice_uint8 = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                inv_gamma = 1.0 / params['gamma']
                table = np.array([((i / 255.0) ** inv_gamma) * 255 
                                for i in np.arange(0, 256)]).astype(np.uint8)
                slice_2d = cv2.LUT(slice_uint8, table).astype(np.float32)
            
            elif params['name'] == 'Bilateral':
                slice_uint8 = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                slice_2d = cv2.bilateralFilter(
                    slice_uint8,
                    d=params['d'],
                    sigmaColor=params['sigma_color'],
                    sigmaSpace=params['sigma_space']
                ).astype(np.float32)
            
            elif params['name'] == 'Gaussian':
                slice_uint8 = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                slice_2d = cv2.GaussianBlur(slice_uint8, (0, 0), params['sigma']).astype(np.float32)
            
            elif params['name'] == 'Median':
                slice_uint8 = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                ksize = params['ksize']
                if ksize % 2 == 0:
                    ksize += 1
                slice_2d = cv2.medianBlur(slice_uint8, ksize).astype(np.float32)
            
            elif params['name'] == 'Shock':
                slice_2d = self.apply_shock_filter(slice_2d, params['iterations'])
            
            elif params['name'] == 'Ridge':
                slice_2d = self.apply_ridge_enhancement(slice_2d, params['threshold'], params['boost'])
            
            elif params['name'] == 'Speckle':
                slice_uint8 = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                slice_2d = cv2.fastNlMeansDenoising(
                    slice_uint8,
                    h=params['h'],
                    templateWindowSize=params['template'],
                    searchWindowSize=params['search']
                ).astype(np.float32)
            
            elif params['name'] == 'Unsharp':
                slice_uint8 = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                blurred = cv2.GaussianBlur(slice_uint8, (0, 0), params['sigma'])
                slice_2d = cv2.addWeighted(
                    slice_uint8, 1.0 + params['amount'],
                    blurred, -params['amount'], 0
                ).astype(np.float32)
            
            elif params['name'] == 'Directional':
                slice_2d = self.apply_directional_filter(slice_2d, params['n_orientations'], params['sigma'])
            
            result[:, :, z] = slice_2d
        
        return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def apply_shock_filter(self, image: np.ndarray, iterations: int) -> np.ndarray:
        result = image.copy()
        for _ in range(iterations):
            norm_img = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            grad_x = cv2.Scharr(norm_img, cv2.CV_32F, 1, 0)
            grad_y = cv2.Scharr(norm_img, cv2.CV_32F, 0, 1)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            laplacian = cv2.Laplacian(norm_img, cv2.CV_32F)
            shock = np.sign(laplacian) * grad_mag
            result = result + 0.05 * shock
            result = np.clip(result, 0, 255)
        return result
    
    def apply_ridge_enhancement(self, image: np.ndarray, threshold: float, boost: float) -> np.ndarray:
        img_norm = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        grad_x = cv2.Scharr(img_norm, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(img_norm, cv2.CV_32F, 0, 1)
        Ixx = cv2.Sobel(grad_x, cv2.CV_32F, 1, 0, ksize=3)
        Iyy = cv2.Sobel(grad_y, cv2.CV_32F, 0, 1, ksize=3)
        Ixy = cv2.Sobel(grad_x, cv2.CV_32F, 0, 1, ksize=3)
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        discriminant = np.sqrt(np.maximum((Ixx - Iyy)**2 + 4*Ixy**2, 0))
        lambda_max = (trace + discriminant) / 2
        lambda_min = (trace - discriminant) / 2
        ridge_strength = np.maximum(np.abs(lambda_max), np.abs(lambda_min))
        ridge_strength = cv2.normalize(ridge_strength, None, 0, 1, cv2.NORM_MINMAX)
        ridge_mask = ridge_strength > threshold
        enhanced = image.copy()
        enhanced[ridge_mask] = np.clip(enhanced[ridge_mask] * boost, 0, 255)
        return enhanced
    
    def apply_directional_filter(self, image: np.ndarray, n_orientations: int, sigma: float) -> np.ndarray:
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_x = cv2.Sobel(img_norm, cv2.CV_32F, 1, 0)
        grad_y = cv2.Sobel(img_norm, cv2.CV_32F, 0, 1)
        theta_map = np.arctan2(grad_y, grad_x)
        
        filtered_responses = []
        for i in range(n_orientations):
            theta = i * np.pi / n_orientations
            kernel = cv2.getGaborKernel((9, 9), sigma, theta, 5.0, 0.5)
            filtered = cv2.filter2D(img_norm.astype(np.float32), cv2.CV_32F, kernel)
            filtered_responses.append(filtered)
        
        angle_indices = ((theta_map + np.pi) / (2 * np.pi) * n_orientations).astype(int)
        angle_indices = np.clip(angle_indices, 0, n_orientations - 1)
        
        result = np.zeros_like(img_norm, dtype=np.float32)
        for i in range(n_orientations):
            mask = (angle_indices == i)
            result[mask] = filtered_responses[i][mask]
        
        return result
    
    def reset(self):
        """Reset to original"""
        self.current_volume = self.original_volume.copy()
        self.filter_list.clear_filters()
        self.update_image()
        
        if hasattr(self.parent(), 'parent') and hasattr(self.parent().parent(), 'update_final_blend'):
            self.parent().parent().update_final_blend()
    
    def update_image(self):
        """Update the displayed image - NO TRANSPOSE"""
        slice_2d = self.current_volume[:, :, self.z_slice_idx]
        
        height, width = slice_2d.shape
        slice_norm = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        slice_norm = np.ascontiguousarray(slice_norm)
        bytes_per_line = width
        
        q_img = QImage(slice_norm.tobytes(), width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
    
    def get_current_volume(self) -> np.ndarray:
        return self.current_volume


class PyramidTunerGUI(QMainWindow):
    """
    Compact GUI for 1080x1920 screens
    """
    
    def __init__(self, volume_3d: np.ndarray):
        super().__init__()
        
        self.original_volume = volume_3d.copy()
        
        print("Building pyramid...")
        self.pyramid_levels = self.build_pyramid(volume_3d)
        
        self.blend_weights = [0.50, 0.30, 0.15, 0.05]
        
        self.setup_ui()
        self.update_final_blend()
    
    def build_pyramid(self, volume: np.ndarray) -> List[np.ndarray]:
        """Build 4-level Gaussian pyramid"""
        vol_norm = cv2.normalize(volume, None, 0, 255, cv2.NORM_MINMAX)
        vol_uint8 = vol_norm.astype(np.uint8)
        
        pyramid = [vol_uint8]
        current = vol_uint8
        
        for level in range(1, 4):
            temp_down = cv2.pyrDown(current[:, :, 0])
            new_x, new_y = temp_down.shape
            z_size = current.shape[2]
            
            downsampled_xy = np.zeros((new_x, new_y, z_size), dtype=np.uint8)
            
            for z in range(z_size):
                downsampled_xy[:, :, z] = cv2.pyrDown(current[:, :, z])
            
            if z_size > 1:
                z_new = z_size // 2
                downsampled_xyz = np.zeros((new_x, new_y, z_new), dtype=np.uint8)
                
                for z in range(z_new):
                    if 2*z + 1 < z_size:
                        downsampled_xyz[:, :, z] = (
                            downsampled_xy[:, :, 2*z].astype(np.uint16) + 
                            downsampled_xy[:, :, 2*z + 1].astype(np.uint16)
                        ) // 2
                    else:
                        downsampled_xyz[:, :, z] = downsampled_xy[:, :, 2*z]
                
                current = downsampled_xyz.astype(np.uint8)
            else:
                current = downsampled_xy
            
            pyramid.append(current)
        
        return pyramid
    
    def setup_ui(self):
        """Compact UI for 1080x1920"""
        self.setWindowTitle("Pyramid Tuner - Compact Mode")
        self.setGeometry(50, 50, 1850, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Use splitter for resizable sections
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top: 4 pyramid levels (horizontal)
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setSpacing(5)
        
        self.level_panels = []
        for i in range(4):
            panel = PyramidLevelPanel(i, self.pyramid_levels[i], top_widget)
            self.level_panels.append(panel)
            top_layout.addWidget(panel)
        
        top_widget.setLayout(top_layout)
        main_splitter.addWidget(top_widget)
        
        # Bottom: Final result + controls (compact)
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout()
        bottom_layout.setSpacing(5)
        
        # Final image
        image_layout = QHBoxLayout()
        self.final_image_label = QLabel()
        self.final_image_label.setFixedSize(400, 400)
        self.final_image_label.setScaledContents(True)
        self.final_image_label.setFrameStyle(QFrame.Shape.Box)
        image_layout.addWidget(self.final_image_label)
        
        # Controls next to image
        controls_layout = QVBoxLayout()
        
        # Z slider
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.final_z_slider = QSlider(Qt.Orientation.Horizontal)
        self.final_z_slider.setMinimum(0)
        self.final_z_slider.setMaximum(self.original_volume.shape[2] - 1)
        self.final_z_slider.setValue(self.original_volume.shape[2] // 2)
        self.final_z_slider.valueChanged.connect(self.update_final_image_only)
        z_layout.addWidget(self.final_z_slider)
        self.final_z_label = QLabel(str(self.final_z_slider.value()))
        z_layout.addWidget(self.final_z_label)
        controls_layout.addLayout(z_layout)
        
        # Blend weights
        controls_layout.addWidget(QLabel("Blend Weights:"))
        weights_layout = QHBoxLayout()
        self.weight_spins = []
        for i in range(4):
            w_layout = QVBoxLayout()
            w_layout.addWidget(QLabel(f"L{i}:"))
            w_spin = QDoubleSpinBox()
            w_spin.setRange(0.0, 1.0)
            w_spin.setValue(self.blend_weights[i])
            w_spin.setSingleStep(0.05)
            w_spin.setDecimals(2)
            w_spin.valueChanged.connect(self.on_weight_changed)
            self.weight_spins.append(w_spin)
            w_layout.addWidget(w_spin)
            weights_layout.addLayout(w_layout)
        
        normalize_btn = QPushButton("Normalize")
        normalize_btn.clicked.connect(self.normalize_weights)
        weights_layout.addWidget(normalize_btn)
        controls_layout.addLayout(weights_layout)
        
        # Action buttons (compact)
        action_layout = QVBoxLayout()
        
        save_btn = QPushButton("Save Final")
        save_btn.clicked.connect(self.save_final_result)
        action_layout.addWidget(save_btn)
        
        save_all_btn = QPushButton("Save All Levels")
        save_all_btn.clicked.connect(self.save_all_levels)
        action_layout.addWidget(save_all_btn)
        
        controls_layout.addLayout(action_layout)
        controls_layout.addStretch()
        
        image_layout.addLayout(controls_layout)
        bottom_layout.addLayout(image_layout)
        
        bottom_widget.setLayout(bottom_layout)
        main_splitter.addWidget(bottom_widget)
        
        # Set splitter proportions (60% top, 40% bottom)
        main_splitter.setStretchFactor(0, 6)
        main_splitter.setStretchFactor(1, 4)
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        
        central_widget.setLayout(main_layout)
        
        self.final_blend = None
    
    def on_weight_changed(self):
        for i, spin in enumerate(self.weight_spins):
            self.blend_weights[i] = spin.value()
        self.update_final_blend()
    
    def normalize_weights(self):
        total = sum(self.blend_weights)
        if total > 0:
            self.blend_weights = [w / total for w in self.blend_weights]
            for i, w in enumerate(self.blend_weights):
                self.weight_spins[i].blockSignals(True)
                self.weight_spins[i].setValue(w)
                self.weight_spins[i].blockSignals(False)
            self.update_final_blend()
    
    def update_final_blend(self):
        print("Updating final blend...")
        
        current_levels = [panel.get_current_volume() for panel in self.level_panels]
        
        weights = np.array(self.blend_weights)
        weights = weights / weights.sum()
        
        target_shape = self.original_volume.shape
        result = np.zeros(target_shape, dtype=np.float32)
        
        for level, (vol, weight) in enumerate(zip(current_levels, weights)):
            upsampled = self.upsample_to_target(vol, target_shape, level)
            result += weight * upsampled
        
        self.final_blend = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        print("Final blend complete")
        self.update_final_image_only()
    
    def update_final_image_only(self):
        """Update final image - NO TRANSPOSE"""
        if self.final_blend is None:
            return
        
        z_idx = self.final_z_slider.value()
        self.final_z_label.setText(str(z_idx))
        
        slice_2d = self.final_blend[:, :, z_idx]
        
        height, width = slice_2d.shape
        slice_2d = np.ascontiguousarray(slice_2d)
        bytes_per_line = width
        
        q_img = QImage(slice_2d.tobytes(), width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
        self.final_image_label.setPixmap(pixmap)
    
    def upsample_to_target(self, volume: np.ndarray, target_shape: tuple, level: int) -> np.ndarray:
        upsampled_xy = np.zeros((target_shape[0], target_shape[1], volume.shape[2]), dtype=np.float32)
        
        for z in range(volume.shape[2]):
            slice_2d = volume[:, :, z].astype(np.float32)
            
            for _ in range(level):
                slice_2d = cv2.pyrUp(slice_2d)
            
            if slice_2d.shape[:2] != target_shape[:2]:
                slice_2d = cv2.resize(slice_2d, (target_shape[1], target_shape[0]))
            
            upsampled_xy[:, :, z] = slice_2d
        
        if upsampled_xy.shape[2] != target_shape[2]:
            upsampled_xyz = np.zeros(target_shape, dtype=np.float32)
            
            for x in range(target_shape[0]):
                for y in range(target_shape[1]):
                    z_old = np.arange(upsampled_xy.shape[2])
                    z_new = np.linspace(0, upsampled_xy.shape[2] - 1, target_shape[2])
                    upsampled_xyz[x, y, :] = np.interp(z_new, z_old, upsampled_xy[x, y, :])
            
            return upsampled_xyz
        else:
            return upsampled_xy
    
    def save_final_result(self):
        if self.final_blend is None:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Final Result", "", "NIfTI Files (*.nii.gz)"
        )
        
        if filepath:
            affine = np.eye(4)
            nii_img = nib.Nifti1Image(self.final_blend, affine)
            nib.save(nii_img, filepath)
            print(f"Saved: {filepath}")
    
    def save_all_levels(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if folder:
            folder_path = Path(folder)
            
            for i, panel in enumerate(self.level_panels):
                filepath = folder_path / f"level_{i}_enhanced.nii.gz"
                affine = np.eye(4)
                nii_img = nib.Nifti1Image(panel.get_current_volume(), affine)
                nib.save(nii_img, filepath)
                print(f"Saved Level {i}: {filepath}")
            
            if self.final_blend is not None:
                filepath = folder_path / "final_blended.nii.gz"
                affine = np.eye(4)
                nii_img = nib.Nifti1Image(self.final_blend, affine)
                nib.save(nii_img, filepath)
                print(f"Saved Final: {filepath}")


def image_tuner(volume_3d: np.ndarray, jupyter: bool = False):
    """
    Launch pyramid tuner
    
    Args:
        volume_3d: 3D volume (X, Y, Z)
        jupyter: True if in Jupyter notebook
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = PyramidTunerGUI(volume_3d)
    window.show()
    
    if jupyter:
        return window
    else:
        if app.instance().thread() != threading.current_thread():
            sys.exit(app.exec())
        return window