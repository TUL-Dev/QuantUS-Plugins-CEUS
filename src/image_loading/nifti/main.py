import numpy as np
import nibabel as nib

from ..transforms import resample_to_spacing
from ...data_objs.image import UltrasoundImage

class EntryClass(UltrasoundImage):
    """
    Loader class for NIfTI CEUS image data.

    This class parses CEUS data from NIfTI files, extracting pixel data, pixel dimensions,
    frame rate for the scan.
    The following attributes are set:
        - pixel_data, pixdim, frame_rate: for the scan
    Output pixel data is in uint8 format with (sagittal, coronal, axial, time) dimensions.

    Kwargs:
        - resample_spacing: tuple of (z, y, x) spacing in mm to resample the image to.
        - transpose: whether to transpose the pixel data (default False).
    """
    required_kwargs = ['resample_spacing', 'transpose']
    extensions = [".nii", ".nii.gz"]
    spatial_dims = 3
    
    def __init__(self, scan_path: str, **kwargs):
        super().__init__(scan_path)
        
        # Supported file extensions for this loader
        assert max([scan_path.endswith(x) for x in self.extensions]), f"File must end with {self.extensions}"
        
        img = nib.load(scan_path)
        header = img.header
        pixdim = header.get_zooms()[:3]  # tuple of pixel dimensions
        frame_rate = 1.0 / header.get_zooms()[3] if len(header.get_zooms()) > 3 else None

        if kwargs.get('transpose', False):
            self.pixel_data = np.asarray(img.dataobj, dtype=np.uint8).T
        else:
            self.pixel_data = np.asarray(img.dataobj, dtype=np.uint8)

        if kwargs.get('resample_spacing', None):
            resample_spacing = kwargs['resample_spacing']
            if len(resample_spacing) != 3:
                raise ValueError("resample_spacing must be a tuple of (z, y, x) spacing in mm.")
            self.pixel_data = resample_to_spacing(self.pixel_data, pixdim, resample_spacing, is_label=False)
            self.resampled_pixdim = resample_spacing

        self.pixdim = pixdim
        self.frame_rate = frame_rate
        self.intensities_for_analysis = self.pixel_data
