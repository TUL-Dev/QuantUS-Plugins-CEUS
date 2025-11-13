"""
Philips 3D/4D CEUS Volume Reconstruction Script - Batch Processing Version
Processes all volumes with progress bar and saves as pickle files

Author: Ported to Python
Original MATLAB code by: F. Quivira, S. Wang, DPD
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
import pickle
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp


class PhilipsVolumeReconstructor:
    """Class for reading and reconstructing Philips SIP volume data"""
    
    def __init__(self):
        self.params = {}
        self.img_data = {}
        self.sc_params = {}
        self.linVol = None
        self.nlinVol = None
        self.destFolder = None
        
    def read_vdb_params(self, filename):
        """
        Read scan conversion VDB parameters from text file
        
        Args:
            filename: Path to the VDB parameters file (e.g., *_vdbDump.xml_Extras.txt)
        
        Returns:
            Dictionary of parameters
        """
        print('Reading SIP scan conversion VDB params...')
        params = {}
        
        with open(filename, 'r') as fid:
            for line in fid:
                line = line.strip()
                if ' = ' in line:
                    param_name, param_value = line.split(' = ', 1)
                    # Remove brackets
                    param_value = param_value.strip('[]').strip()
                    
                    # Handle values with metadata (e.g., '1.325e+01 , Pset 0')
                    if ',' in param_value:
                        param_value = param_value.split(',')[0].strip()
                    
                    try:
                        # Try to evaluate as number
                        params[param_name] = float(param_value)
                    except:
                        # Keep as string if can't convert
                        params[param_name] = param_value
        
        print('Finished reading SIP scan conversion VDB params...')
        return params
    
    def read_sip_interleaved_3d(self, filename, number_of_planes=32, 
                                number_of_params=5, max_frames=10000):
        """
        Read interleaved SIP volume data (linear and non-linear streams)
        
        Args:
            filename: Path to the .raw SIP file
            number_of_planes: Number of elevation planes per volume
            number_of_params: Number of header parameters (5 or 7)
            max_frames: Maximum number of frames to read
        
        Returns:
            Dictionary containing linear and non-linear volumes
        """
        print('Reading interleaved SIP volume data...')
        
        img = {
            'linImage': [],
            'nlinImage': [],
            'linVol': [],
            'nlinVol': []
        }
        param = {
            'image_pitch': [],
            'number_lines': [],
            'number_focal_zone': [],
            'number_lateral_multiline': [],
            'number_elevation_multiline': []
        }
        
        with open(filename, 'rb') as fid:
            frame_idx = 0
            
            while frame_idx < max_frames:
                try:
                    # Read header parameters (uint32)
                    header = np.frombuffer(fid.read(4 * number_of_params), 
                                         dtype=np.uint32)
                    
                    if len(header) < number_of_params:
                        break
                    
                    if number_of_params == 5:
                        param['image_pitch'].append(header[0])
                        param['number_lines'].append(header[1])
                        param['number_focal_zone'].append(header[2])
                        param['number_lateral_multiline'].append(header[3])
                        param['number_elevation_multiline'].append(header[4])
                    elif number_of_params == 7:
                        param['image_pitch'].append(header[0])
                        param['number_lines'].append(header[1])
                        param['number_focal_zone'].append(header[3])
                        param['number_lateral_multiline'].append(header[5])
                        param['number_elevation_multiline'].append(header[6])
                    
                    # Read image data (uint16)
                    num_samples = param['image_pitch'][frame_idx] // 2
                    num_lines = param['number_lines'][frame_idx]
                    total_pixels = num_samples * num_lines
                    
                    linebuf = np.frombuffer(fid.read(2 * total_pixels), 
                                          dtype=np.uint16)
                    
                    if len(linebuf) < total_pixels:
                        break
                    
                    linebuf = linebuf.reshape((num_samples, num_lines), order='F')
                    
                    # Deinterleave linear and non-linear data
                    img['linImage'].append(linebuf[0::2, :])
                    img['nlinImage'].append(linebuf[1::2, :])
                    
                    frame_idx += 1
                    
                except Exception as e:
                    print(f'Reached EOF or error: {e}')
                    break
        
        print(f'Read {frame_idx} frames')
        
        # Reshape into volumes
        total_num_frames = len(img['linImage'])
        total_num_frames_full_vol = total_num_frames - (total_num_frames % number_of_planes)
        num_volumes = total_num_frames_full_vol // number_of_planes
        
        print(f'Total frames: {total_num_frames}, Complete volumes: {num_volumes}')
        
        if num_volumes > 0:
            # Get dimensions from first frame
            nz, nx = img['linImage'][0].shape
            
            # Pre-allocate arrays
            img['linVol'] = np.zeros((num_volumes, nz, nx, number_of_planes), dtype=np.uint16)
            img['nlinVol'] = np.zeros((num_volumes, nz, nx, number_of_planes), dtype=np.uint16)
            
            for vol_idx in range(num_volumes):
                for plane_idx in range(number_of_planes):
                    frame_idx = vol_idx * number_of_planes + plane_idx
                    img['linVol'][vol_idx, :, :, plane_idx] = img['linImage'][frame_idx]
                    img['nlinVol'][vol_idx, :, :, plane_idx] = img['nlinImage'][frame_idx]
        
        print('Finished reading interleaved SIP volume data...')
        return img, param
    
    def scan_convert_3d_va(self, rx_lines, line_angles, plane_angles, 
                          beam_dist, img_size, fov_size, z0):
        """
        Convert 3D image from polar to Cartesian coordinates using reverse interpolation
        with virtual apex geometry
        
        Args:
            rx_lines: Array with scan line data (depth, lines, planes)
            line_angles: Azimuthal steering angles in degrees
            plane_angles: Elevation steering angles in degrees
            beam_dist: Axial distance array in mm
            img_size: Size in pixels of output image [x, y, z]
            fov_size: Size in mm of output image [x, y, z]
            z0: Virtual apex distance (z offset) in mm
        
        Returns:
            img: Cartesian image
            x_loc, y_loc, z_loc: Coordinate vectors
        """
        # Create Cartesian grid
        pix_size_x = 1 / (img_size[0] * 2 - 1)  # Lateral
        pix_size_y = 1 / (img_size[1] - 1)      # Elevation
        pix_size_z = 1 / (img_size[2] * 4 - 1)  # Axial Depth
        
        x_loc = (np.arange(0, 1 + pix_size_x, pix_size_x) - 0.5) * fov_size[0]
        y_loc = (np.arange(0, 1 + pix_size_y, pix_size_y) - 0.5) * fov_size[1]
        z_loc = np.arange(0, 1 + pix_size_z, pix_size_z) * fov_size[2]
        
        # Create meshgrid
        Z, X, Y = np.meshgrid(z_loc, x_loc, y_loc, indexing='ij')
        
        # Virtual apex geometry - case 3 (curved array)
        PHI = np.arctan2(Y, Z + z0)
        TH = np.arctan2(X, np.sqrt(Y**2 + (Z + z0)**2))
        R = np.sqrt(X**2 + Y**2 + (Z + z0)**2) * (1 - z0 / np.sqrt(Y**2 + (Z + z0)**2))
        
        # Convert angles to radians
        rad_line_angles = np.deg2rad(line_angles)
        rad_plane_angles = np.deg2rad(plane_angles)
        
        # Perform interpolation
        points = (beam_dist, rad_line_angles, rad_plane_angles)
        coords = np.stack([R.ravel(), TH.ravel(), PHI.ravel()], axis=-1)
        
        # Interpolate
        img_flat = interpn(points, rx_lines, coords, 
                          method='linear', bounds_error=False, fill_value=np.nan)
        
        img = img_flat.reshape(R.shape)
        
        return img, x_loc, y_loc, z_loc
    
    def enhance_bmode_image(self, volume, gamma=2.0, denoise_sigma=0.5):
        """
        Apply image enhancement to B-mode volume
        
        Args:
            volume: 3D volume array
            gamma: Gamma correction value (default=2.0)
            denoise_sigma: Gaussian filter sigma for denoising (default=0.5)
        
        Returns:
            Enhanced volume
        """
        # Normalize to [0, 255]
        vol_normalized = (volume - 3e4) * 255 / 3e4
        vol_normalized = np.clip(vol_normalized, 0, 255)
        
        # Apply gamma correction
        vol_gamma = np.power(vol_normalized / 255.0, gamma) * 255
        
        # Apply light denoising
        if denoise_sigma > 0:
            vol_enhanced = gaussian_filter(vol_gamma, sigma=denoise_sigma)
        else:
            vol_enhanced = vol_gamma
        
        return vol_enhanced.astype(np.uint8)
    
    def format_volume_pix(self, unformatted_volume):
        """
        Format volume for saving (transpose to match expected orientation)
        
        Args:
            unformatted_volume: Input volume
        
        Returns:
            Formatted volume
        """
        unformatted_volume = np.array(unformatted_volume).squeeze()
        unformatted_volume = np.transpose(unformatted_volume.swapaxes(0, 1))
        return unformatted_volume
    
    def prep_vol_read(self, path_to_data, sip_filename, dest_folder, pix_per_mm=1.2):
        """
        Prepare for volume reading by loading parameters and SIP data
        
        Args:
            path_to_data: Path to data directory
            sip_filename: SIP filename
            dest_folder: Destination folder for outputs
            pix_per_mm: Pixel per mm resolution
        
        Returns:
            Destination folder path
        """
        # Get VDB filename
        vdb_filename = "_".join(sip_filename.split("_")[:2]) + "_vdbDump.xml"
        sc_param_filename = vdb_filename + "_Extras.txt"
        
        # Read parameters
        self.sc_params = self.read_vdb_params(
            os.path.join(path_to_data, sc_param_filename)
        )
        
        # Add required parameters
        if 'NUM_PLANES' not in self.sc_params:
            self.sc_params['NUM_PLANES'] = 20
        if 'PixPerMm' not in self.sc_params:
            self.sc_params['PixPerMm'] = pix_per_mm
        if 'VDB_THREED_RT_VOLUME_RATE' not in self.sc_params:
            self.sc_params['VDB_THREED_RT_VOLUME_RATE'] = 0.0
        
        # Read SIP data
        sip_vol_dat, _ = self.read_sip_interleaved_3d(
            os.path.join(path_to_data, sip_filename),
            number_of_planes=int(self.sc_params['NUM_PLANES'])
        )
        
        self.linVol = sip_vol_dat['linVol']
        self.nlinVol = sip_vol_dat['nlinVol']
        
        # Create destination folder
        self.destFolder = Path(dest_folder)
        dest_folder_name = "_".join(sip_filename.split("_")[:2])
        self.destFolder = self.destFolder / Path(dest_folder_name)
        self.destFolder.mkdir(exist_ok=True, parents=True)
        
        return self.destFolder
    
    def save_single_vol(self, vol_indices):
        """
        Process and save volumes for given indices
        
        Args:
            vol_indices: List of volume indices to process
        
        Returns:
            Tuple of (bmode_dims, ceus_dims, bmode_shape, ceus_shape)
        """
        for vol_index in tqdm(vol_indices, desc='Processing volumes'):
            # Scan convert linear volume (B-mode)
            lin_vol_sc, fov_size = self.scan_convert_volume_series_single(
                self.linVol[vol_index], 
                self.sc_params, 
                is_lin=True
            )
            
            # Scan convert non-linear volume (CEUS)
            nlin_vol_sc, _ = self.scan_convert_volume_series_single(
                self.nlinVol[vol_index], 
                self.sc_params, 
                is_lin=False
            )
            
            # Apply enhancement to B-mode
            lin_vol_enhanced = self.enhance_bmode_image(lin_vol_sc, gamma=2.0, denoise_sigma=0.5)
            
            # Normalize CEUS
            nlin_vol_normalized = (nlin_vol_sc - 3.5e4) * 255 / 1.7e4
            nlin_vol_normalized = np.clip(nlin_vol_normalized, 0, 255).astype(np.uint8)
            
            # Format volumes
            lin_vol_formatted = self.format_volume_pix(lin_vol_enhanced)
            nlin_vol_formatted = self.format_volume_pix(nlin_vol_normalized)
            
            # Save as pickle files
            with open(self.destFolder / f"bmode_frame_{vol_index}.pkl", 'wb') as f:
                pickle.dump(lin_vol_formatted, f)
            
            with open(self.destFolder / f"ceus_frame_{vol_index}.pkl", 'wb') as f:
                pickle.dump(nlin_vol_formatted, f)
        
        # Calculate dimensions for return
        bmode_dims = [fov_size[2], fov_size[0], fov_size[1]]  # [Axial, Lateral, Elevation]
        ceus_dims = [fov_size[2], fov_size[0], fov_size[1]]
        bmode_shape = lin_vol_formatted.shape
        ceus_shape = nlin_vol_formatted.shape
        
        return bmode_dims, ceus_dims, bmode_shape, ceus_shape
    
    def scan_convert_volume_series_single(self, volume, sc_params, is_lin=True):
        """
        Scan convert a single volume
        
        Args:
            volume: 3D volume
            sc_params: Scan conversion parameters
            is_lin: Whether this is linear (B-mode) or non-linear (CEUS)
        
        Returns:
            Tuple of (converted_volume, fov_size)
        """
        nz, nx, ny = volume.shape
        
        # Helper function to ensure numeric type
        def to_float(val):
            if isinstance(val, str):
                val = val.split(',')[0].strip()
                return float(val)
            return float(val)
        
        # Extract parameters
        apex_dist = to_float(sc_params['VDB_2D_ECHO_APEX_TO_SKINLINE'])
        azim_start = np.rad2deg(to_float(sc_params['VDB_2D_ECHO_START_WIDTH_GC']))
        azim_end = np.rad2deg(to_float(sc_params['VDB_2D_ECHO_STOP_WIDTH_GC']))
        rx_ang_az = np.linspace(azim_start, azim_end, nx)
        
        elev_start = np.rad2deg(to_float(sc_params['VDB_THREED_START_ELEVATION_ACTUAL']))
        elev_end = np.rad2deg(to_float(sc_params['VDB_THREED_STOP_ELEVATION_ACTUAL']))
        rx_ang_el = np.linspace(elev_start, elev_end, ny)
        
        depth_mm = to_float(sc_params['VDB_2D_ECHO_STOP_DEPTH_SIP'])
        img_dpth = np.linspace(0, depth_mm, nz)
        
        start_depth = to_float(sc_params['VDB_2D_ECHO_START_DEPTH_SIP'])
        
        # Calculate FOV
        vol_depth = depth_mm * (abs(np.sin(np.deg2rad(elev_start))) + 
                               abs(np.sin(np.deg2rad(elev_end))))
        vol_width = depth_mm * (abs(np.sin(np.deg2rad(azim_start))) + 
                               abs(np.sin(np.deg2rad(azim_end))))
        vol_height = depth_mm - start_depth
        
        fov_size = [vol_width, vol_depth, vol_height]
        
        # Image size
        pix_per_mm = to_float(sc_params.get('PixPerMm', 1.2))
        img_size = [int(np.round(pix_per_mm * vol_width)),
                   int(np.round(pix_per_mm * vol_depth)),
                   int(np.round(pix_per_mm * vol_height))]
        
        # Scan convert
        img, _, _, _ = self.scan_convert_3d_va(
            volume, rx_ang_az, rx_ang_el, img_dpth,
            img_size, fov_size, apex_dist
        )
        
        return img, fov_size


def sip_parser(data_folder, dest_folder, sip_filename, n_procs, pix_per_mm):
    """
    Main parser function with multiprocessing support
    
    Args:
        data_folder: Source data folder
        dest_folder: Destination folder
        sip_filename: SIP filename
        n_procs: Number of processes
        pix_per_mm: Pixel per mm resolution
    """
    # Initialize parser
    philips_parser = PhilipsVolumeReconstructor()
    vol_dest_path = philips_parser.prep_vol_read(data_folder, sip_filename, dest_folder, pix_per_mm)
    
    print(f"\nTotal volumes to process: {philips_parser.linVol.shape[0]}")
    
    # Split volume indices for multiprocessing
    vol_inds = list(range(philips_parser.linVol.shape[0]))
    split_inds = np.array_split(vol_inds, n_procs)
    
    # Create processes
    procs = []
    for ind_chunk in split_inds:
        proc = mp.Process(target=philips_parser.save_single_vol, args=(ind_chunk,))
        procs.append(proc)
    
    # Start processes
    for proc in procs:
        proc.start()
    
    # Wait for completion
    for proc in procs:
        proc.join()
    
    # Get dimensions from first volume
    bmode_dims, ceus_dims, bmode_shape, ceus_shape = philips_parser.save_single_vol([0])
    
    # Save dimension info
    timeconst = philips_parser.sc_params['VDB_THREED_RT_VOLUME_RATE']
    bmode_res = [4., bmode_dims[0]/bmode_shape[0], bmode_dims[1]/bmode_shape[1], 
                bmode_dims[2]/bmode_shape[2], timeconst, 0., 0., 0.] # [Axial, Lateral, Elevation]
    ceus_res = [4., ceus_dims[0]/ceus_shape[0], ceus_dims[1]/ceus_shape[1], 
               ceus_dims[2]/ceus_shape[2], timeconst, 0., 0., 0.]
    
    with open(vol_dest_path / "bmode_volume_dims.pkl", 'wb') as f:
        pickle.dump(bmode_res, f)
    
    with open(vol_dest_path / "ceus_volume_dims.pkl", 'wb') as f:
        pickle.dump(ceus_res, f)
    
    print(f"\nProcessing complete! Output saved to: {vol_dest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Philips SIP Volume Parser')
    parser.add_argument('dataFolder', type=str, help='Parent folder of file to parse')
    parser.add_argument('destFolder', type=str, help='Destination folder of outputs')
    parser.add_argument('sipFilename', type=str, help='Name of file to parse')
    parser.add_argument('nProcs', type=int, help='Number of processes for parsing')
    parser.add_argument('pixPerMm', type=float, help='Resolution of output volumes')
    
    args = parser.parse_args()
    
    sip_parser(args.dataFolder, args.destFolder, args.sipFilename, args.nProcs, args.pixPerMm)