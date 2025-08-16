"""
VOI (Volume of Interest) Loading View for MVC architecture
"""

import os
from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QFileDialog, QWidget, QHBoxLayout, QDialog
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal

from quantus.gui.mvc.base_view import BaseViewMixin
from quantus.gui.seg_loading.voi_selection_ui import Ui_constructRoi
from quantus.data_objs import UltrasoundRfImage, BmodeSeg


class VoiLoadingView(QDialog, Ui_constructRoi, BaseViewMixin):
    """
    View for VOI (Volume of Interest) loading interface for 3D images.
    
    Handles user interface for 3D segmentation type selection, file selection,
    VOI drawing, and segmentation preview following the existing QuantUS GUI style.
    """
    
    # Signals for communicating with controller
    user_action = pyqtSignal(str, object)  # action_name, action_data
    close_requested = pyqtSignal()
    back_requested = pyqtSignal()
    
    def __init__(self, image_data: UltrasoundRfImage, parent: Optional[QWidget] = None):
        QDialog.__init__(self, parent)
        self.__init_base_view__(parent)
        self._image_data = image_data
        self._file_extensions: list = []
        self._loading_widgets: list = []
        self._matplotlib_canvas: Optional[FigureCanvas] = None
        self._current_voi_coords: Optional[tuple] = None
        
    def setup_ui(self) -> None:
        """Setup the user interface from compiled .ui file."""
        self.setupUi(self)
        
        # Apply layout configuration matching original VOI implementation
        self.setLayout(self.full_screen_layout)
        self.full_screen_layout.removeItem(self.voi_layout)
        self._hide_voi_layout()
        self.full_screen_layout.removeItem(self.seg_loading_layout)
        self._hide_seg_loading_layout()
        self.full_screen_layout.setStretchFactor(self.side_bar_layout, 1)
        self.full_screen_layout.setStretchFactor(self.select_type_layout, 10)
        
        # Store widgets that should be disabled during loading
        self._loading_widgets = [
            self.seg_type_dropdown,
            self.accept_type_button,
            self.choose_seg_path_button,
            self.clear_seg_path_button,
            self.load_seg_button,
            self.draw_voi_button,
            self.confirm_voi_button,
            self.back_button,
            self.confirm_seg_button,
            self.home_button
        ]
        
        # Initialize visibility
        self._show_type_selection_layout()
        
        # Setup matplotlib canvas for VOI drawing
        self._setup_matplotlib_canvas()
        
    def _setup_matplotlib_canvas(self) -> None:
        """Setup matplotlib canvas for 3D volume display and VOI drawing."""
        if hasattr(self, 'volume_frame_widget'):
            # Create matplotlib figure and canvas for 3D volume visualization
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.patch.set_facecolor('white')
            self._matplotlib_canvas = FigureCanvas(fig)
            
            # Add canvas to the volume frame widget
            layout = QHBoxLayout(self.volume_frame_widget)
            layout.addWidget(self._matplotlib_canvas)
            self.volume_frame_widget.setLayout(layout)
            
    def connect_signals(self) -> None:
        """Connect UI signals to internal handlers."""
        self.accept_type_button.clicked.connect(self._on_type_accepted)
        self.choose_seg_path_button.clicked.connect(self._on_choose_seg_path)
        self.clear_seg_path_button.clicked.connect(self.seg_path_input.clear)
        self.load_seg_button.clicked.connect(self._on_load_segmentation)
        self.draw_voi_button.clicked.connect(self._on_draw_voi)
        self.confirm_voi_button.clicked.connect(self._on_confirm_voi)
        self.back_button.clicked.connect(self._on_back_clicked)
        self.confirm_seg_button.clicked.connect(self._on_confirm_segmentation)
        
    def update_display(self, data: Any) -> None:
        """
        Update the view with new data.
        
        Args:
            data: BmodeSeg data or None
        """
        if isinstance(data, BmodeSeg):
            # Segmentation loading completed successfully
            self._show_segmentation_preview(data)
            self._show_seg_confirmation_layout()
            
    def set_seg_loaders(self, loader_names: list) -> None:
        """
        Set available segmentation loaders in the dropdown.
        
        Args:
            loader_names: List of formatted segmentation loader names
        """
        self.seg_type_dropdown.clear()
        self.seg_type_dropdown.addItems(loader_names)
        
    def show_loading(self) -> None:
        """Show loading state in the UI."""
        super().show_loading()
        for widget in self._loading_widgets:
            widget.setEnabled(False)
            
    def hide_loading(self) -> None:
        """Hide loading state in the UI."""
        super().hide_loading()
        for widget in self._loading_widgets:
            widget.setEnabled(True)
            
    def show_error(self, error_message: str) -> None:
        """
        Display error message to user.
        
        Args:
            error_message: Error message to display
        """
        if hasattr(self, 'select_seg_error_msg'):
            self.select_seg_error_msg.setText(error_message)
            self.select_seg_error_msg.show()
        else:
            print(f"VOI Error: {error_message}")
            
    def clear_error(self) -> None:
        """Clear any displayed error message."""
        if hasattr(self, 'select_seg_error_msg'):
            self.select_seg_error_msg.clear()
            self.select_seg_error_msg.hide()
            
    def _on_type_accepted(self) -> None:
        """Handle segmentation type selection acceptance."""
        selected_type = self.seg_type_dropdown.currentText()
        if selected_type:
            self.clear_error()
            self._emit_user_action('seg_type_selected', selected_type)
            
    def _on_choose_seg_path(self) -> None:
        """Handle segmentation file selection."""
        self._select_file_helper(self.seg_path_input, self._file_extensions)
        
    def _on_load_segmentation(self) -> None:
        """Handle segmentation loading request."""
        seg_path = self.seg_path_input.text().strip()
        
        if not seg_path:
            self.show_error("Please select a segmentation file")
            return
            
        self.clear_error()
        
        # Get additional kwargs if needed
        seg_loader_kwargs = {}
        
        self._emit_user_action('load_segmentation', {
            'seg_path': seg_path,
            'seg_loader_kwargs': seg_loader_kwargs
        })
        
    def _on_draw_voi(self) -> None:
        """Handle VOI drawing request."""
        self._show_voi_layout()
        self._display_volume_for_voi()
        
    def _on_confirm_voi(self) -> None:
        """Handle VOI confirmation."""
        if self._current_voi_coords:
            self._emit_user_action('voi_created', self._current_voi_coords)
        else:
            self.show_error("Please draw a VOI first")
            
    def _on_confirm_segmentation(self) -> None:
        """Handle segmentation confirmation."""
        self._emit_user_action('segmentation_confirmed', None)
        
    def _on_back_clicked(self) -> None:
        """Handle back button click."""
        # Determine current state and go back appropriately
        self._show_type_selection_layout()
        self._hide_seg_loading_layout()
        self._hide_voi_layout()
        self._hide_seg_confirmation_layout()
        self.clear_error()
        self.back_requested.emit()
        
    def _select_file_helper(self, path_input, file_exts: list) -> None:
        """
        Helper method for file selection dialogs.
        
        Args:
            path_input: QLineEdit widget to update with selected path
            file_exts: List of file extensions for filtering
        """
        # Check if file path is manually typed and exists
        if os.path.exists(path_input.text()):
            return
            
        # Create filter string
        if file_exts:
            filter_str = " ".join([f"*{ext}" for ext in file_exts])
        else:
            filter_str = "All Files (*)"
            
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open File", 
            filter=filter_str
        )
        
        if file_name:
            path_input.setText(file_name)
            
    def show_seg_selection(self) -> None:
        """Show the segmentation loading interface."""
        self._hide_type_selection_layout()
        self._show_seg_loading_layout()
        
    def _display_volume_for_voi(self) -> None:
        """Display 3D volume for VOI drawing."""
        if not self._matplotlib_canvas:
            return
            
        try:
            # Get 3D volume data for display
            fig = self._matplotlib_canvas.figure
            fig.clear()
            
            # Create subplots for different views (axial, sagittal, coronal)
            ax_axial = fig.add_subplot(2, 2, 1)
            ax_sagittal = fig.add_subplot(2, 2, 2)
            ax_coronal = fig.add_subplot(2, 2, 3)
            ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
            
            # Display placeholder volume data
            # In real implementation, you would convert the 3D ultrasound data to displayable format
            placeholder_volume = np.random.rand(50, 50, 50)  # Placeholder 3D volume
            
            # Show different slices
            mid_slice = placeholder_volume.shape[2] // 2
            ax_axial.imshow(placeholder_volume[:, :, mid_slice], cmap='gray')
            ax_axial.set_title('Axial View')
            
            ax_sagittal.imshow(placeholder_volume[:, mid_slice, :], cmap='gray')
            ax_sagittal.set_title('Sagittal View')
            
            ax_coronal.imshow(placeholder_volume[mid_slice, :, :], cmap='gray')
            ax_coronal.set_title('Coronal View')
            
            # 3D visualization (simplified)
            x, y, z = np.meshgrid(range(10), range(10), range(10))
            ax_3d.scatter(x, y, z, alpha=0.1)
            ax_3d.set_title('3D View')
            
            plt.tight_layout()
            self._matplotlib_canvas.draw()
            
        except Exception as e:
            self.show_error(f"Error displaying volume: {e}")
            
    def _show_segmentation_preview(self, seg_data: BmodeSeg) -> None:
        """
        Show preview of loaded 3D segmentation.
        
        Args:
            seg_data: Loaded segmentation data
        """
        try:
            if hasattr(self, 'preview_frame_widget') and self._matplotlib_canvas:
                # Display 3D segmentation overlay
                fig = self._matplotlib_canvas.figure
                fig.clear()
                ax = fig.add_subplot(111, projection='3d')
                
                # Display segmentation mask (simplified 3D visualization)
                if hasattr(seg_data, 'seg_mask'):
                    # Simple 3D visualization of segmentation
                    ax.set_title("3D Segmentation Preview")
                    self._matplotlib_canvas.draw()
                    
        except Exception as e:
            self.show_error(f"Error showing segmentation preview: {e}")
            
    # Layout visibility methods (specific to VOI layout)
    def _hide_type_selection_layout(self) -> None:
        """Hide the segmentation type selection layout."""
        self.select_type_label.hide()
        self.seg_type_dropdown.hide()
        self.accept_type_button.hide()
        
    def _show_type_selection_layout(self) -> None:
        """Show the segmentation type selection layout."""
        self.select_type_label.show()
        self.seg_type_dropdown.show()
        self.accept_type_button.show()
        
    def _hide_seg_loading_layout(self) -> None:
        """Hide the segmentation loading layout."""
        if hasattr(self, 'load_seg_button'):
            self.load_seg_button.hide()
        if hasattr(self, 'choose_seg_path_button'):
            self.choose_seg_path_button.hide()
        if hasattr(self, 'seg_path_input'):
            self.seg_path_input.hide()
        if hasattr(self, 'draw_voi_button'):
            self.draw_voi_button.hide()
            
    def _show_seg_loading_layout(self) -> None:
        """Show the segmentation loading layout."""
        self.full_screen_layout.addItem(self.seg_loading_layout)
        if hasattr(self, 'load_seg_button'):
            self.load_seg_button.show()
        if hasattr(self, 'choose_seg_path_button'):
            self.choose_seg_path_button.show()
        if hasattr(self, 'seg_path_input'):
            self.seg_path_input.show()
        if hasattr(self, 'draw_voi_button'):
            self.draw_voi_button.show()
            
    def _hide_voi_layout(self) -> None:
        """Hide the VOI drawing layout."""
        if hasattr(self, 'confirm_voi_button'):
            self.confirm_voi_button.hide()
        # Hide matplotlib canvas if needed
            
    def _show_voi_layout(self) -> None:
        """Show the VOI drawing layout."""
        self.full_screen_layout.addItem(self.voi_layout)
        if hasattr(self, 'confirm_voi_button'):
            self.confirm_voi_button.show()
            
    def _hide_seg_confirmation_layout(self) -> None:
        """Hide the segmentation confirmation layout."""
        if hasattr(self, 'confirm_seg_button'):
            self.confirm_seg_button.hide()
        if hasattr(self, 'home_button'):
            self.home_button.hide()
            
    def _show_seg_confirmation_layout(self) -> None:
        """Show the segmentation confirmation layout."""
        if hasattr(self, 'confirmation_layout'):
            self.full_screen_layout.addItem(self.confirmation_layout)
        if hasattr(self, 'confirm_seg_button'):
            self.confirm_seg_button.show()
        if hasattr(self, 'home_button'):
            self.home_button.show()
