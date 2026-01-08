# Simple Napari 3D Ultrasound Viewer - Memory Safe
# Add this to your notebook

import napari
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from IPython.display import display


def simple_napari_viewer(image_data, seg_data=None, frame_number = 0):
    """
    Simple Napari viewer that avoids memory issues by not pre-processing all frames
    """
    viewer = napari.Viewer(ndisplay=3, title="3D Ultrasound Viewer")
    
    # Add contrast data (original, no enhancement to save memory)
    if image_data is not None:
        contrast_data = image_data
        contrast_data = contrast_data[:, :, :]
        print(f"Adding contrast data: {contrast_data.shape}")
        
        # Transpose to (T, Z, Y, X) for napari
        contrast_transposed = np.transpose(contrast_data, (0, 1, 2))
        
        viewer.add_image(
            contrast_transposed,
            name='Contrast',
            colormap='gray',
            opacity=0.8
        )
    
    # Add segmentation
    if seg_data is not None:
        seg_mask = seg_data.seg_mask
        seg_mask = seg_mask[:, :, :, frame_number]
        print(f"Adding segmentation: {seg_mask.shape}")
        
        if seg_mask.ndim == 4:  # (Z, Y, X, T)
            seg_transposed = np.transpose(seg_mask, (0, 1, 2))  # (T, Z, Y, X)
        else:  # (Z, Y, X) - single frame
            seg_transposed = seg_mask[np.newaxis, ...]  # Add time dimension
        
        viewer.add_labels(
            seg_transposed,
            name='Segmentation',
            opacity=0.5
        )
    
    # Set 3D view
    viewer.dims.ndisplay = 3
    
    print("\n✅ Napari viewer launched!")
    print("🎮 Controls:")
    print("   - Mouse wheel: Zoom")
    print("   - Left drag: Rotate 3D view") 
    print("   - Time slider: Navigate frames")
    print("   - Layer controls: Adjust contrast/opacity")
    
    return viewer

def interactive_ortho_viewer(bmode_image_data):
    """
    Interactive orthogonal viewer using sliders to control crosshair position
    This will work in any Jupyter environment
    """
    
    def plot_orthogonal_views(lateral_slice, depth_slice, elevation_slice):
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # === AXIAL VIEW ===
        axial_img = np.transpose(bmode_image_data[:, :, elevation_slice])
        axes[0].imshow(axial_img, cmap='gray')
        axes[0].axhline(y=depth_slice, color='red', linewidth=2, alpha=0.8)
        axes[0].axvline(x=lateral_slice, color='red', linewidth=2, alpha=0.8)
        axes[0].set_title(f'Axial (Z={elevation_slice})')
        axes[0].set_xlabel('Lateral (X)')
        axes[0].set_ylabel('Depth (Y)')
        
        # === CORONAL VIEW ===
        coronal_img = bmode_image_data[:, depth_slice, :]
        axes[1].imshow(coronal_img, cmap='gray')
        axes[1].axhline(y=elevation_slice, color='red', linewidth=2, alpha=0.8)
        axes[1].axvline(x=lateral_slice, color='red', linewidth=2, alpha=0.8)
        axes[1].set_title(f'Coronal (Y={depth_slice})')
        axes[1].set_xlabel('Elevation (Z)')
        axes[1].set_ylabel('Lateral (X)')
        
        # === SAGITTAL VIEW ===
        sagittal_img = bmode_image_data[lateral_slice, :, :]
        axes[2].imshow(sagittal_img, cmap='gray')
        axes[2].axhline(y=elevation_slice, color='red', linewidth=2, alpha=0.8)
        axes[2].axvline(x=depth_slice, color='red', linewidth=2, alpha=0.8)
        axes[2].set_title(f'Sagittal (X={lateral_slice})')
        axes[2].set_xlabel('Elevation (Z)')
        axes[2].set_ylabel('Depth (Y)')
        
        plt.tight_layout()
        plt.show()
    
    # Create interactive sliders
    lateral_slider = IntSlider(
        value=bmode_image_data.shape[0]//2, 
        min=0, 
        max=bmode_image_data.shape[0]-1, 
        step=1,
        description='Lateral (X):'
    )
    
    depth_slider = IntSlider(
        value=bmode_image_data.shape[1]//2, 
        min=0, 
        max=bmode_image_data.shape[1]-1, 
        step=1,
        description='Depth (Y):'
    )
    
    elevation_slider = IntSlider(
        value=bmode_image_data.shape[2]//2, 
        min=0, 
        max=bmode_image_data.shape[2]-1, 
        step=1,
        description='Elevation (Z):'
    )
    
    # Create interactive plot
    interact(plot_orthogonal_views, 
             lateral_slice=lateral_slider,
             depth_slice=depth_slider, 
             elevation_slice=elevation_slider)
    
    print(f"Data shape: {bmode_image_data.shape}")
    print("Use the sliders above to move the crosshairs!")

