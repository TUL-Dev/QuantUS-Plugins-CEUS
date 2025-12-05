"""
GPU-Accelerated Chunked 3D Scan Conversion for Philips Ultrasound Data

Features:
1. Pre-compute coordinate arrays once (R, TH, PHI)
2. Chunked processing along Z-axis to limit memory usage
3. GPU acceleration using CuPy and custom CUDA kernels
4. Fallback to CPU if GPU unavailable
5. Drop-in replacement for existing scanConvert3Va function

Usage:
    from scanconvert_gpu_chunked import GPUScanConverter
    
    converter = GPUScanConverter(sc_params, resolution_scale=4.0)
    converter.precompute_geometry(n_samples, n_lines, n_planes)
    
    # Convert single frame
    output = converter.convert_frame(rx_lines)
    
    # Convert all frames
    outputs = converter.convert_series(rx_lines_series)
"""

import numpy as np
import scipy.interpolate
from dataclasses import dataclass
from typing import Tuple, Optional, List
from tqdm import tqdm
import gc

# GPU imports with fallback
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    from numba import cuda
    import math
    HAS_NUMBA_CUDA = cuda.is_available() if hasattr(cuda, 'is_available') else False
except ImportError:
    HAS_NUMBA_CUDA = False


def get_gpu_memory_info():
    """Get available GPU memory in bytes"""
    if HAS_CUPY:
        try:
            mempool = cp.get_default_memory_pool()
            free, total = cp.cuda.Device().mem_info
            return {'free': free, 'total': total, 'used': mempool.used_bytes()}
        except:
            return None
    return None


@dataclass
class ScanGeometry:
    """Container for pre-computed scan conversion geometry"""
    # Output dimensions
    img_size: Tuple[int, int, int]  # (nx, ny, nz) - Cartesian output
    fov_size: Tuple[float, float, float]  # (width, depth, height) in mm
    output_shape: Tuple[int, int, int]  # Actual output array shape
    
    # Pre-computed coordinate arrays (can be chunked)
    R: np.ndarray
    TH: np.ndarray  
    PHI: np.ndarray
    
    # Input polar grid coordinates
    beam_dist: np.ndarray
    rad_line_angles: np.ndarray
    rad_plane_angles: np.ndarray
    
    # Apex distance
    z0: float


class GPUScanConverter:
    """
    GPU-accelerated scan converter with chunked processing.
    
    Parameters
    ----------
    sc_params : ScParams
        Scan conversion parameters from VDB file
    resolution_scale : float
        Multiply output resolution by this factor (e.g., 4.0 for 4× resolution)
    chunk_size_z : int
        Number of Z slices to process at once (controls GPU memory usage)
    use_gpu : bool
        Whether to use GPU acceleration (auto-detected if None)
    gpu_memory_limit_gb : float
        Maximum GPU memory to use (in GB)
    """
    
    def __init__(
        self,
        sc_params,
        resolution_scale: float = 1.0,
        chunk_size_z: int = 64,
        use_gpu: bool = True,
        gpu_memory_limit_gb: float = 4.0
    ):
        self.sc_params = sc_params
        self.resolution_scale = resolution_scale
        self.chunk_size_z = chunk_size_z
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        
        # Determine if GPU is available
        self.use_gpu = use_gpu and (HAS_CUPY or HAS_NUMBA_CUDA)
        if use_gpu and not self.use_gpu:
            print("Warning: GPU requested but not available. Using CPU.")
        
        self.geometry: Optional[ScanGeometry] = None
        self._gpu_geometry_cached = False
        
        # GPU arrays (cached)
        self._d_beam_dist = None
        self._d_rad_line_angles = None
        self._d_rad_plane_angles = None
        
    def precompute_geometry(
        self,
        n_samples: int,
        n_lines: int,
        n_planes: int,
        verbose: bool = True
    ) -> ScanGeometry:
        """
        Pre-compute all coordinate arrays for scan conversion.
        Call this ONCE before converting any frames.
        
        Parameters
        ----------
        n_samples : int
            Number of depth samples in raw data (nz)
        n_lines : int
            Number of azimuth lines in raw data (nx)
        n_planes : int
            Number of elevation planes in raw data (ny)
        verbose : bool
            Print progress information
        """
        if verbose:
            print("=" * 60)
            print("Pre-computing scan conversion geometry...")
            print(f"  Resolution scale: {self.resolution_scale}×")
            print(f"  GPU enabled: {self.use_gpu}")
            print("=" * 60)
        
        # Extract parameters
        z0 = self.sc_params.VDB_2D_ECHO_APEX_TO_SKINLINE
        azim_start = self.sc_params.VDB_2D_ECHO_START_WIDTH_GC * 180 / np.pi
        azim_end = self.sc_params.VDB_2D_ECHO_STOP_WIDTH_GC * 180 / np.pi
        elev_start = self.sc_params.VDB_THREED_START_ELEVATION_ACTUAL * 180 / np.pi
        elev_end = self.sc_params.VDB_THREED_STOP_ELEVATION_ACTUAL * 180 / np.pi
        depth_mm = self.sc_params.VDB_2D_ECHO_STOP_DEPTH_SIP
        
        # Input grid coordinates
        beam_dist = np.linspace(0, depth_mm, n_samples).astype(np.float32)
        rx_ang_az = np.linspace(azim_start, azim_end, n_lines)
        rx_ang_el = np.linspace(elev_start, elev_end, n_planes)
        rad_line_angles = np.deg2rad(rx_ang_az).astype(np.float32)
        rad_plane_angles = np.deg2rad(rx_ang_el).astype(np.float32)
        
        # Calculate FOV
        vol_width = depth_mm * (abs(np.sin(np.deg2rad(azim_start))) + 
                                abs(np.sin(np.deg2rad(azim_end))))
        vol_depth = depth_mm * (abs(np.sin(np.deg2rad(elev_start))) + 
                                abs(np.sin(np.deg2rad(elev_end))))
        vol_height = (self.sc_params.VDB_2D_ECHO_STOP_DEPTH_SIP - 
                      self.sc_params.VDB_2D_ECHO_START_DEPTH_SIP)
        fov_size = (vol_width, vol_depth, vol_height)
        
        # Output image size
        pix_per_mm = self.sc_params.pixPerMm * self.resolution_scale
        img_size = tuple(int(round(dim * pix_per_mm)) for dim in fov_size)
        
        if verbose:
            print(f"  FOV size: {vol_width:.1f} × {vol_depth:.1f} × {vol_height:.1f} mm")
            print(f"  Output size: {img_size[0]} × {img_size[1]} × {img_size[2]} voxels")
        
        # Pre-compute Cartesian grid coordinates
        pix_size_x = 1 / (img_size[0] - 1) if img_size[0] > 1 else 1
        pix_size_y = 1 / (img_size[1] - 1) if img_size[1] > 1 else 1
        pix_size_z = 1 / (img_size[2] - 1) if img_size[2] > 1 else 1
        
        x_loc = ((np.arange(0, 1 + pix_size_x/2, pix_size_x) - 0.5) * fov_size[0]).astype(np.float32)
        y_loc = ((np.arange(0, 1 + pix_size_y/2, pix_size_y) - 0.5) * fov_size[1]).astype(np.float32)
        z_loc = (np.arange(0, 1 + pix_size_z/2, pix_size_z) * fov_size[2]).astype(np.float32)
        
        # Truncate to exact size
        x_loc = x_loc[:img_size[0]]
        y_loc = y_loc[:img_size[1]]
        z_loc = z_loc[:img_size[2]]
        
        if verbose:
            print(f"  Creating coordinate arrays...")
        
        # Create meshgrid and convert to spherical
        # Shape will be (nz, nx, ny) to match original code's indexing='ij'
        Z, X, Y = np.meshgrid(z_loc, x_loc, y_loc, indexing='ij')
        
        # Convert to spherical coordinates
        PHI = np.arctan2(Y, Z + z0).astype(np.float32)
        TH = np.arctan2(X, np.sqrt(Y**2 + (Z + z0)**2)).astype(np.float32)
        R = (np.sqrt(X**2 + Y**2 + (Z + z0)**2) * 
             (1 - z0 / np.sqrt(Y**2 + (Z + z0)**2))).astype(np.float32)
        
        # Free intermediate arrays
        del X, Y, Z
        gc.collect()
        
        output_shape = R.shape
        
        mem_gb = (R.nbytes + TH.nbytes + PHI.nbytes) / 1e9
        if verbose:
            print(f"  Coordinate arrays memory: {mem_gb:.2f} GB")
            print(f"  Output shape: {output_shape}")
        
        self.geometry = ScanGeometry(
            img_size=img_size,
            fov_size=fov_size,
            output_shape=output_shape,
            R=R,
            TH=TH,
            PHI=PHI,
            beam_dist=beam_dist,
            rad_line_angles=rad_line_angles,
            rad_plane_angles=rad_plane_angles,
            z0=z0
        )
        
        # Auto-adjust chunk size based on GPU memory
        if self.use_gpu:
            self._auto_adjust_chunk_size(verbose)
            self._cache_gpu_grid_arrays()
        
        if verbose:
            print(f"  Chunk size (Z): {self.chunk_size_z}")
            print("Geometry pre-computation complete!")
            print("=" * 60)
        
        return self.geometry
    
    def _auto_adjust_chunk_size(self, verbose: bool = True):
        """Automatically adjust chunk size based on available GPU memory"""
        if not self.use_gpu:
            return
            
        gpu_info = get_gpu_memory_info()
        if gpu_info is None:
            return
            
        available_gb = gpu_info['free'] / 1e9
        
        # Estimate memory per chunk:
        # - Input data chunk (float32): chunk_z × nx × ny × 4 bytes
        # - 3 coordinate arrays (float32): 3 × chunk_z × nx × ny × 4 bytes
        # - Output chunk (float32): chunk_z × nx × ny × 4 bytes
        # Total: 5 × chunk_z × nx × ny × 4 bytes
        
        nz, nx, ny = self.geometry.output_shape
        bytes_per_z_slice = 5 * nx * ny * 4  # 5 arrays, float32
        
        # Use at most gpu_memory_limit_gb or 80% of available memory
        max_memory = min(self.gpu_memory_limit_gb * 1e9, available_gb * 0.8 * 1e9)
        
        max_chunk_z = int(max_memory / bytes_per_z_slice)
        max_chunk_z = max(16, min(max_chunk_z, nz))  # Clamp to reasonable range
        
        self.chunk_size_z = min(self.chunk_size_z, max_chunk_z)
        
        if verbose:
            print(f"  GPU memory available: {available_gb:.1f} GB")
            print(f"  Auto-adjusted chunk size: {self.chunk_size_z}")
    
    def _cache_gpu_grid_arrays(self):
        """Cache the 1D grid arrays on GPU (small, reusable)"""
        if not self.use_gpu or not HAS_CUPY:
            return
            
        self._d_beam_dist = cp.asarray(self.geometry.beam_dist)
        self._d_rad_line_angles = cp.asarray(self.geometry.rad_line_angles)
        self._d_rad_plane_angles = cp.asarray(self.geometry.rad_plane_angles)
        self._gpu_geometry_cached = True
    
    def convert_frame(
        self,
        rx_lines: np.ndarray,
        output: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Scan convert a single frame using chunked GPU processing.
        
        Parameters
        ----------
        rx_lines : np.ndarray
            Raw data for single frame, shape (n_samples, n_lines, n_planes)
        output : np.ndarray, optional
            Pre-allocated output array to avoid allocation overhead
            
        Returns
        -------
        np.ndarray
            Scan-converted volume in Cartesian coordinates
        """
        if self.geometry is None:
            raise RuntimeError("Must call precompute_geometry() first!")
        
        # Allocate output if not provided
        if output is None:
            output = np.zeros(self.geometry.output_shape, dtype=np.float32)
        
        if self.use_gpu and HAS_CUPY:
            self._convert_frame_gpu_chunked(rx_lines, output)
        else:
            self._convert_frame_cpu_chunked(rx_lines, output)
        
        return output
    
    def _convert_frame_gpu_chunked(
        self,
        rx_lines: np.ndarray,
        output: np.ndarray
    ):
        """GPU chunked conversion using CuPy"""
        geom = self.geometry
        nz_out = geom.output_shape[0]
        
        # Transfer input data to GPU once
        d_rx_lines = cp.asarray(rx_lines.astype(np.float32))
        
        # Process in chunks along Z axis
        for z_start in range(0, nz_out, self.chunk_size_z):
            z_end = min(z_start + self.chunk_size_z, nz_out)
            
            # Get coordinate chunks (already float32)
            R_chunk = geom.R[z_start:z_end]
            TH_chunk = geom.TH[z_start:z_end]
            PHI_chunk = geom.PHI[z_start:z_end]
            
            # Transfer coordinate chunks to GPU
            d_R = cp.asarray(R_chunk)
            d_TH = cp.asarray(TH_chunk)
            d_PHI = cp.asarray(PHI_chunk)
            
            # Perform GPU interpolation
            chunk_result = self._gpu_trilinear_interp(
                d_rx_lines, d_R, d_TH, d_PHI
            )
            
            # Copy result back to output
            output[z_start:z_end] = cp.asnumpy(chunk_result)
            
            # Free GPU memory for this chunk
            del d_R, d_TH, d_PHI, chunk_result
            cp.get_default_memory_pool().free_all_blocks()
        
        # Free input data
        del d_rx_lines
        cp.get_default_memory_pool().free_all_blocks()
    
    def _gpu_trilinear_interp(
        self,
        d_data: 'cp.ndarray',
        d_R: 'cp.ndarray',
        d_TH: 'cp.ndarray', 
        d_PHI: 'cp.ndarray'
    ) -> 'cp.ndarray':
        """
        Perform trilinear interpolation on GPU using vectorized CuPy operations.
        
        This is a fully vectorized implementation that avoids explicit CUDA kernels.
        """
        geom = self.geometry
        
        # Grid parameters
        r_min, r_max = self._d_beam_dist[0], self._d_beam_dist[-1]
        th_min, th_max = self._d_rad_line_angles[0], self._d_rad_line_angles[-1]
        phi_min, phi_max = self._d_rad_plane_angles[0], self._d_rad_plane_angles[-1]
        
        nr = len(geom.beam_dist)
        nth = len(geom.rad_line_angles)
        nphi = len(geom.rad_plane_angles)
        
        # Calculate grid spacing
        dr = (r_max - r_min) / (nr - 1)
        dth = (th_max - th_min) / (nth - 1)
        dphi = (phi_max - phi_min) / (nphi - 1)
        
        # Compute continuous indices
        r_idx = (d_R - r_min) / dr
        th_idx = (d_TH - th_min) / dth
        phi_idx = (d_PHI - phi_min) / dphi
        
        # Integer indices (floor)
        i0 = cp.floor(r_idx).astype(cp.int32)
        j0 = cp.floor(th_idx).astype(cp.int32)
        k0 = cp.floor(phi_idx).astype(cp.int32)
        
        # Clamp to valid range for indexing
        i0 = cp.clip(i0, 0, nr - 2)
        j0 = cp.clip(j0, 0, nth - 2)
        k0 = cp.clip(k0, 0, nphi - 2)
        
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # Fractional parts (weights)
        wr = cp.clip(r_idx - i0, 0, 1)
        wth = cp.clip(th_idx - j0, 0, 1)
        wphi = cp.clip(phi_idx - k0, 0, 1)
        
        # Create bounds mask
        in_bounds = (
            (r_idx >= 0) & (r_idx <= nr - 1) &
            (th_idx >= 0) & (th_idx <= nth - 1) &
            (phi_idx >= 0) & (phi_idx <= nphi - 1)
        )
        
        # Fetch the 8 corner values
        c000 = d_data[i0, j0, k0]
        c001 = d_data[i0, j0, k1]
        c010 = d_data[i0, j1, k0]
        c011 = d_data[i0, j1, k1]
        c100 = d_data[i1, j0, k0]
        c101 = d_data[i1, j0, k1]
        c110 = d_data[i1, j1, k0]
        c111 = d_data[i1, j1, k1]
        
        # Trilinear interpolation
        # Interpolate along phi (k) axis
        c00 = c000 * (1 - wphi) + c001 * wphi
        c01 = c010 * (1 - wphi) + c011 * wphi
        c10 = c100 * (1 - wphi) + c101 * wphi
        c11 = c110 * (1 - wphi) + c111 * wphi
        
        # Interpolate along theta (j) axis
        c0 = c00 * (1 - wth) + c01 * wth
        c1 = c10 * (1 - wth) + c11 * wth
        
        # Interpolate along r (i) axis
        result = c0 * (1 - wr) + c1 * wr
        
        # Apply bounds mask (set out-of-bounds to 0)
        result = cp.where(in_bounds, result, 0.0)
        
        return result.astype(cp.float32)
    
    def _convert_frame_cpu_chunked(
        self,
        rx_lines: np.ndarray,
        output: np.ndarray
    ):
        """CPU chunked conversion using scipy.interpolate.interpn"""
        geom = self.geometry
        nz_out = geom.output_shape[0]
        
        for z_start in range(0, nz_out, self.chunk_size_z):
            z_end = min(z_start + self.chunk_size_z, nz_out)
            
            # Get coordinate chunks
            R_chunk = geom.R[z_start:z_end]
            TH_chunk = geom.TH[z_start:z_end]
            PHI_chunk = geom.PHI[z_start:z_end]
            
            # Interpolate
            chunk_result = scipy.interpolate.interpn(
                (geom.beam_dist, geom.rad_line_angles, geom.rad_plane_angles),
                rx_lines,
                (R_chunk, TH_chunk, PHI_chunk),
                method='linear',
                bounds_error=False,
                fill_value=0
            )
            
            output[z_start:z_end] = chunk_result.astype(np.float32)
    
    def convert_series(
        self,
        rx_lines_series: np.ndarray,
        scale_params: Optional[dict] = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Scan convert a series of volumes.
        
        Parameters
        ----------
        rx_lines_series : np.ndarray
            Raw data, shape (n_frames, n_samples, n_lines, n_planes)
        scale_params : dict, optional
            Scaling parameters {'threshold': float, 'divisor': float}
        show_progress : bool
            Whether to show progress bar
            
        Returns
        -------
        np.ndarray
            Scan-converted volume series, shape (n_frames, nz, nx, ny)
        """
        if self.geometry is None:
            raise RuntimeError("Must call precompute_geometry() first!")
        
        n_frames = rx_lines_series.shape[0]
        output_shape = (n_frames,) + self.geometry.output_shape
        
        print(f"Converting {n_frames} frames...")
        print(f"Output shape: {output_shape}")
        print(f"Output memory: {np.prod(output_shape) * 4 / 1e9:.2f} GB")
        
        # Allocate output
        output = np.zeros(output_shape, dtype=np.float32)
        
        # Pre-allocate single frame buffer to avoid repeated allocation
        frame_buffer = np.zeros(self.geometry.output_shape, dtype=np.float32)
        
        iterator = tqdm(range(n_frames)) if show_progress else range(n_frames)
        
        for i in iterator:
            # Convert frame (reusing buffer)
            frame_buffer.fill(0)
            self.convert_frame(rx_lines_series[i], output=frame_buffer)
            
            # Apply scaling if requested
            if scale_params:
                frame_buffer = ((frame_buffer - scale_params['threshold']) * 
                               255 / scale_params['divisor'])
            
            output[i] = frame_buffer
        
        return output


# =============================================================================
# Drop-in replacement functions for existing code
# =============================================================================

def scanConvert3Va_gpu(
    rxLines: np.ndarray,
    lineAngles: np.ndarray,
    planeAngles: np.ndarray,
    beamDist: np.ndarray,
    imgSize: Tuple[int, int, int],
    fovSize: Tuple[float, float, float],
    z0: float,
    chunk_size: int = 64
) -> np.ndarray:
    """
    GPU-accelerated drop-in replacement for scanConvert3Va.
    
    Same interface as original function but uses GPU acceleration.
    """
    # Create temporary geometry
    nz, nx, ny = rxLines.shape
    
    # Pre-compute coordinates
    pix_size_x = 1 / (imgSize[0] - 1) if imgSize[0] > 1 else 1
    pix_size_y = 1 / (imgSize[1] - 1) if imgSize[1] > 1 else 1
    pix_size_z = 1 / (imgSize[2] - 1) if imgSize[2] > 1 else 1
    
    x_loc = ((np.arange(0, 1 + pix_size_x/2, pix_size_x) - 0.5) * fovSize[0]).astype(np.float32)
    y_loc = ((np.arange(0, 1 + pix_size_y/2, pix_size_y) - 0.5) * fovSize[1]).astype(np.float32)
    z_loc = (np.arange(0, 1 + pix_size_z/2, pix_size_z) * fovSize[2]).astype(np.float32)
    
    x_loc = x_loc[:imgSize[0]]
    y_loc = y_loc[:imgSize[1]]
    z_loc = z_loc[:imgSize[2]]
    
    rad_line_angles = np.deg2rad(lineAngles).astype(np.float32)
    rad_plane_angles = np.deg2rad(planeAngles).astype(np.float32)
    
    output = np.zeros((len(z_loc), len(x_loc), len(y_loc)), dtype=np.float32)
    nz_out = len(z_loc)
    
    if HAS_CUPY:
        # Transfer input to GPU
        d_rx_lines = cp.asarray(rxLines.astype(np.float32))
        d_beam_dist = cp.asarray(beamDist.astype(np.float32))
        d_rad_line_angles = cp.asarray(rad_line_angles)
        d_rad_plane_angles = cp.asarray(rad_plane_angles)
        
        for z_start in range(0, nz_out, chunk_size):
            z_end = min(z_start + chunk_size, nz_out)
            z_chunk = z_loc[z_start:z_end]
            
            # Create meshgrid for this chunk
            Z, X, Y = np.meshgrid(z_chunk, x_loc, y_loc, indexing='ij')
            
            # Convert to spherical
            PHI = np.arctan2(Y, Z + z0).astype(np.float32)
            TH = np.arctan2(X, np.sqrt(Y**2 + (Z + z0)**2)).astype(np.float32)
            R = (np.sqrt(X**2 + Y**2 + (Z + z0)**2) * 
                 (1 - z0 / np.sqrt(Y**2 + (Z + z0)**2))).astype(np.float32)
            
            del X, Y, Z
            
            # Transfer to GPU
            d_R = cp.asarray(R)
            d_TH = cp.asarray(TH)
            d_PHI = cp.asarray(PHI)
            
            # Interpolate on GPU (using vectorized approach)
            chunk_result = _gpu_interp_chunk(
                d_rx_lines, d_beam_dist, d_rad_line_angles, d_rad_plane_angles,
                d_R, d_TH, d_PHI
            )
            
            output[z_start:z_end] = cp.asnumpy(chunk_result)
            
            del d_R, d_TH, d_PHI, chunk_result
            cp.get_default_memory_pool().free_all_blocks()
        
        del d_rx_lines
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU fallback
        Z, X, Y = np.meshgrid(z_loc, x_loc, y_loc, indexing='ij')
        PHI = np.arctan2(Y, Z + z0)
        TH = np.arctan2(X, np.sqrt(Y**2 + (Z + z0)**2))
        R = np.sqrt(X**2 + Y**2 + (Z + z0)**2) * (1 - z0 / np.sqrt(Y**2 + (Z + z0)**2))
        
        output = scipy.interpolate.interpn(
            (beamDist, rad_line_angles, rad_plane_angles),
            rxLines, (R, TH, PHI),
            method='linear', bounds_error=False, fill_value=0
        )
    
    return np.array(output)


def _gpu_interp_chunk(d_data, d_beam_dist, d_rad_line_angles, d_rad_plane_angles,
                      d_R, d_TH, d_PHI):
    """Helper function for GPU interpolation"""
    r_min, r_max = d_beam_dist[0], d_beam_dist[-1]
    th_min, th_max = d_rad_line_angles[0], d_rad_line_angles[-1]
    phi_min, phi_max = d_rad_plane_angles[0], d_rad_plane_angles[-1]
    
    nr = len(d_beam_dist)
    nth = len(d_rad_line_angles)
    nphi = len(d_rad_plane_angles)
    
    dr = (r_max - r_min) / (nr - 1)
    dth = (th_max - th_min) / (nth - 1)
    dphi = (phi_max - phi_min) / (nphi - 1)
    
    r_idx = (d_R - r_min) / dr
    th_idx = (d_TH - th_min) / dth
    phi_idx = (d_PHI - phi_min) / dphi
    
    i0 = cp.clip(cp.floor(r_idx).astype(cp.int32), 0, nr - 2)
    j0 = cp.clip(cp.floor(th_idx).astype(cp.int32), 0, nth - 2)
    k0 = cp.clip(cp.floor(phi_idx).astype(cp.int32), 0, nphi - 2)
    
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1
    
    wr = cp.clip(r_idx - i0, 0, 1)
    wth = cp.clip(th_idx - j0, 0, 1)
    wphi = cp.clip(phi_idx - k0, 0, 1)
    
    in_bounds = (
        (r_idx >= 0) & (r_idx <= nr - 1) &
        (th_idx >= 0) & (th_idx <= nth - 1) &
        (phi_idx >= 0) & (phi_idx <= nphi - 1)
    )
    
    c000 = d_data[i0, j0, k0]
    c001 = d_data[i0, j0, k1]
    c010 = d_data[i0, j1, k0]
    c011 = d_data[i0, j1, k1]
    c100 = d_data[i1, j0, k0]
    c101 = d_data[i1, j0, k1]
    c110 = d_data[i1, j1, k0]
    c111 = d_data[i1, j1, k1]
    
    c00 = c000 * (1 - wphi) + c001 * wphi
    c01 = c010 * (1 - wphi) + c011 * wphi
    c10 = c100 * (1 - wphi) + c101 * wphi
    c11 = c110 * (1 - wphi) + c111 * wphi
    
    c0 = c00 * (1 - wth) + c01 * wth
    c1 = c10 * (1 - wth) + c11 * wth
    
    result = c0 * (1 - wr) + c1 * wr
    result = cp.where(in_bounds, result, 0.0)
    
    return result.astype(cp.float32)


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GPU-Accelerated Chunked Scan Conversion - Test Suite")
    print("=" * 70)
    
    # Check GPU availability
    print(f"\nGPU Status:")
    print(f"  CuPy available: {HAS_CUPY}")
    print(f"  Numba CUDA available: {HAS_NUMBA_CUDA}")
    
    if HAS_CUPY:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"  GPU memory: {gpu_info['free']/1e9:.1f} GB free / {gpu_info['total']/1e9:.1f} GB total")
    
    # Create mock parameters
    class MockScParams:
        def __init__(self):
            self.VDB_2D_ECHO_APEX_TO_SKINLINE = 20.0
            self.VDB_2D_ECHO_START_WIDTH_GC = -0.35
            self.VDB_2D_ECHO_STOP_WIDTH_GC = 0.35
            self.VDB_THREED_START_ELEVATION_ACTUAL = -0.35
            self.VDB_THREED_STOP_ELEVATION_ACTUAL = 0.35
            self.VDB_2D_ECHO_STOP_DEPTH_SIP = 150.0
            self.VDB_2D_ECHO_START_DEPTH_SIP = 0.0
            self.pixPerMm = 1.2
    
    sc_params = MockScParams()
    n_samples, n_lines, n_planes = 500, 128, 32
    
    # Test different resolution scales
    for scale in [1.0, 2.0]:
        print(f"\n{'='*70}")
        print(f"Testing {scale}× resolution")
        print("=" * 70)
        
        converter = GPUScanConverter(
            sc_params,
            resolution_scale=scale,
            chunk_size_z=64,
            use_gpu=HAS_CUPY
        )
        
        converter.precompute_geometry(n_samples, n_lines, n_planes)
        
        # Create test data
        test_data = np.random.rand(n_samples, n_lines, n_planes).astype(np.float32)
        
        # Time the conversion
        import time
        
        start = time.time()
        result = converter.convert_frame(test_data)
        elapsed = time.time() - start
        
        print(f"\nConversion time: {elapsed:.3f} seconds")
        print(f"Output shape: {result.shape}")
        print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
        
        # Compare with CPU if GPU was used
        if converter.use_gpu:
            print("\nComparing GPU vs CPU results...")
            converter_cpu = GPUScanConverter(
                sc_params,
                resolution_scale=scale,
                chunk_size_z=64,
                use_gpu=False
            )
            converter_cpu.precompute_geometry(n_samples, n_lines, n_planes, verbose=False)
            
            start = time.time()
            result_cpu = converter_cpu.convert_frame(test_data)
            elapsed_cpu = time.time() - start
            
            max_diff = np.abs(result - result_cpu).max()
            print(f"CPU time: {elapsed_cpu:.3f} seconds")
            print(f"Speedup: {elapsed_cpu/elapsed:.1f}×")
            print(f"Max difference: {max_diff:.6f}")