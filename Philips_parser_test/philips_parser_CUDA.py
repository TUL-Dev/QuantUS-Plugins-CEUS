"""
Integration Example: GPU-Accelerated Philips 4D Parser

This script shows how to integrate the GPU-accelerated scan conversion
with the existing Philips parser code.

Usage:
    python philips_parser_gpu.py <dataFolder> <destFolder> <sipFilename> <pixPerMm> [--scale 4.0]
"""

import os
import pickle
from pathlib import Path
from typing import Tuple, List
import numpy as np
from tqdm import tqdm
import argparse

# Import the GPU converter
from scanconvert_gpu_chunked import GPUScanConverter, HAS_CUPY

# Import existing parser components (adjust import path as needed)
# from functions import ScParams, SipVolParams, SipVolDataStruct, readSIPscVDBParams, readSIP3dInterleavedV5


class ScParams():
    """Scan conversion parameters from VDB file"""
    def __init__(self):
        self.NUM_PLANES: int = 20
        self.pixPerMm: float = 1.2
        self.VDB_2D_ECHO_APEX_TO_SKINLINE: float = 20.0
        self.VDB_2D_ECHO_START_WIDTH_GC: float = -0.35
        self.VDB_2D_ECHO_STOP_WIDTH_GC: float = 0.35
        self.VDB_THREED_START_ELEVATION_ACTUAL: float = -0.35
        self.VDB_THREED_STOP_ELEVATION_ACTUAL: float = 0.35
        self.VDB_2D_ECHO_STOP_DEPTH_SIP: float = 150.0
        self.VDB_2D_ECHO_START_DEPTH_SIP: float = 0.0
        self.VDB_2D_ECHO_SLACK_TIME_MM: float = 0.0
        self.VDB_THREED_RT_VOLUME_RATE: float = 0.0


def readSIPscVDBParams(filename):
    """Read scan conversion parameters from VDB extras file"""
    print("Reading SIP scan conversion VDB Params...")
    file = open(filename, "r")
    scParams = ScParams()
    for line in file:
        paramName, paramValue = line.split(" = ")
        try: 
            paramValue, _ = paramValue.split(" \n")
        except ValueError:
            paramValue, _ = paramValue.split(" ,")
        paramAr = paramValue.split(" ")
        for i in range(len(paramAr)):
            paramAr[i] = float(paramAr[i])

        if len(paramAr) == 1:
            paramValue = paramAr[0]
        else:
            paramValue = paramAr

        if (paramName == 'VDB_2D_ECHO_MATRIX_ELEVATION_NUM_TRANSMIT_PLANES'):
            scParams.NUM_PLANES = int(paramValue)
        elif (paramName == 'pixPerMm'):
            scParams.pixPerMm = paramValue
        elif (paramName == 'VDB_2D_ECHO_APEX_TO_SKINLINE'):
            scParams.VDB_2D_ECHO_APEX_TO_SKINLINE = paramValue
        elif (paramName == 'VDB_2D_ECHO_START_WIDTH_GC'):
            scParams.VDB_2D_ECHO_START_WIDTH_GC = paramValue
        elif (paramName == 'VDB_2D_ECHO_STOP_WIDTH_GC'):
            scParams.VDB_2D_ECHO_STOP_WIDTH_GC = paramValue
        elif (paramName == 'VDB_THREED_START_ELEVATION_ACTUAL'):
            scParams.VDB_THREED_START_ELEVATION_ACTUAL = paramValue
        elif (paramName == 'VDB_THREED_STOP_ELEVATION_ACTUAL'):
            scParams.VDB_THREED_STOP_ELEVATION_ACTUAL = paramValue
        elif (paramName == 'VDB_2D_ECHO_STOP_DEPTH_SIP'):
            scParams.VDB_2D_ECHO_STOP_DEPTH_SIP = paramValue
        elif (paramName == 'VDB_2D_ECHO_START_DEPTH_SIP'):
            scParams.VDB_2D_ECHO_START_DEPTH_SIP = paramValue
        elif (paramName == 'VDB_2D_ECHO_SLACK_TIME_MM'):
            scParams.VDB_2D_ECHO_SLACK_TIME_MM = paramValue
        elif (paramName == 'VDB_THREED_RT_VOLUME_RATE'):
            scParams.VDB_THREED_RT_VOLUME_RATE = paramValue

    file.close()        
    print('Finished reading SIP scan conversion VDB params...')
    return scParams


def formatVolumePix(unformattedVolume: np.ndarray) -> np.ndarray:
    """Format volume for output (transpose axes)"""
    unformattedVolume = np.array(unformattedVolume).squeeze()
    unformattedVolume = np.transpose(unformattedVolume.swapaxes(0, 1))
    return unformattedVolume


class Philips4dParserGPU:
    """
    GPU-accelerated Philips 4D ultrasound parser.
    
    Key improvements over original:
    1. Pre-computes coordinate arrays ONCE
    2. Uses GPU for interpolation
    3. Processes frames in chunks to manage memory
    4. Avoids multiprocessing overhead (GPU is faster)
    """
    
    def __init__(
        self,
        resolution_scale: float = 1.0,
        chunk_size_z: int = 64,
        use_gpu: bool = True,
        gpu_memory_limit_gb: float = 4.0
    ):
        """
        Parameters
        ----------
        resolution_scale : float
            Multiply output resolution (e.g., 4.0 for 4× higher resolution)
        chunk_size_z : int
            Z-axis chunk size for GPU processing
        use_gpu : bool
            Whether to use GPU acceleration
        gpu_memory_limit_gb : float
            Maximum GPU memory to use
        """
        self.resolution_scale = resolution_scale
        self.chunk_size_z = chunk_size_z
        self.use_gpu = use_gpu and HAS_CUPY
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        
        self.scParams: ScParams = None
        self.destFolder: Path = None
        self.linVol: np.ndarray = None
        self.nLinVol: np.ndarray = None
        
        # GPU converters (initialized after geometry is known)
        self.lin_converter: GPUScanConverter = None
        self.nlin_converter: GPUScanConverter = None
        
    def prepVolRead(self, pathToData: str, sipFilename: str, destFolder: str, pixPerMm: float = 1.2):
        """Read volume data and prepare for processing"""
        # Read VDB parameters
        vdbFilename = str("_".join(sipFilename.split("_")[:2]) + "_vdbDump.xml")
        scParamFilename = str(vdbFilename + "_Extras.txt")
        
        self.scParams = readSIPscVDBParams(os.path.join(pathToData, scParamFilename))
        if not hasattr(self.scParams, 'NUM_PLANES'):
            self.scParams.NUM_PLANES = 20
        if not hasattr(self.scParams, 'pixPerMm'):
            self.scParams.pixPerMm = pixPerMm
        
        # Apply resolution scale to pixPerMm
        self.scParams.pixPerMm = pixPerMm * self.resolution_scale
        
        # Read the raw data
        print(f"Reading raw data from {sipFilename}...")
        self._readSIPData(os.path.join(pathToData, sipFilename))
        
        # Setup destination folder
        self.destFolder = Path(destFolder)
        destFolderName = "_".join(sipFilename.split("_")[:2])
        self.destFolder = self.destFolder / Path(destFolderName)
        self.destFolder.mkdir(exist_ok=True, parents=True)
        
        # Initialize GPU converters with pre-computed geometry
        self._initializeConverters()
        
        return self.destFolder
    
    def _readSIPData(self, filepath: str):
        """Read SIP interleaved volume data"""
        numPlanes = self.scParams.NUM_PLANES
        stpSample = 2
        paramLen = 5
        
        params = np.fromfile(filepath, dtype=np.int32, count=paramLen)
        numSamples = int(params[0] / stpSample)
        numLines = int(params[1])
        numPixels = numSamples * numLines
        
        buffer = np.fromfile(filepath, dtype=np.uint16, count=-1)
        paramOffs = 2 * paramLen
        numSlices = int(buffer.size / (numPixels + paramOffs))
        numVolumes = int(np.floor(numSlices / numPlanes))
        
        print(f"  Found {numVolumes} volumes, each with {numPlanes} planes")
        print(f"  Each plane: {numSamples} samples × {numLines} lines")
        
        out = np.zeros((numSamples, numLines, numPlanes, numVolumes), dtype=np.float32)
        
        for v in tqdm(range(numVolumes), desc="Reading volumes"):
            offs = (numPixels + paramOffs) * numPlanes * v + paramOffs
            offs = int(offs)
            for a in range(numPlanes):
                out[:, :, a, v] = buffer[offs:offs + numPixels].reshape(
                    (numSamples, numLines), order='F'
                )
                offs += numPixels + paramOffs
        
        # Split into linear (B-mode) and non-linear (CEUS)
        nonLinSample = 2
        linSample = 1
        
        self.nLinVol = out[np.arange(nonLinSample - 1, out.shape[0], stpSample)]
        self.linVol = out[np.arange(linSample - 1, out.shape[0], stpSample)]
        
        # Transpose to [nFrames, nz, nx, ny] format
        self.nLinVol = np.transpose(self.nLinVol, (3, 0, 1, 2))
        self.linVol = np.transpose(self.linVol, (3, 0, 1, 2))
        
        print(f"  Linear volume shape: {self.linVol.shape}")
        print(f"  Non-linear volume shape: {self.nLinVol.shape}")
    
    def _initializeConverters(self):
        """Initialize GPU converters with pre-computed geometry"""
        n_samples, n_lines, n_planes = self.linVol.shape[1:4]
        
        print("\nInitializing GPU converters...")
        
        # Create converters (they share the same geometry)
        self.lin_converter = GPUScanConverter(
            self.scParams,
            resolution_scale=1.0,  # Already applied to pixPerMm
            chunk_size_z=self.chunk_size_z,
            use_gpu=self.use_gpu,
            gpu_memory_limit_gb=self.gpu_memory_limit_gb
        )
        
        self.lin_converter.precompute_geometry(n_samples, n_lines, n_planes)
        
        # Non-linear uses same geometry
        self.nlin_converter = self.lin_converter
    
    def processAllVolumes(
        self,
        save_individual: bool = True,
        return_arrays: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Process all volumes with GPU acceleration.
        
        Parameters
        ----------
        save_individual : bool
            Save each frame as individual pickle file
        return_arrays : bool
            Return the full arrays (warning: high memory for many frames)
            
        Returns
        -------
        bmodeDims, ceusDims, bmodeShape, ceusShape
        """
        n_frames = self.linVol.shape[0]
        geom = self.lin_converter.geometry
        
        # Scaling parameters
        lin_scale = {'threshold': 3e4, 'divisor': 3e4}
        nlin_scale = {'threshold': 3.5e4, 'divisor': 1.7e4}
        
        # Get dimensions for metadata
        fov_size = geom.fov_size
        bmodeDims = [fov_size[2], fov_size[0], fov_size[1]]  # [Axial, Lateral, Elevation]
        ceusDims = bmodeDims.copy()
        
        print(f"\nProcessing {n_frames} frames...")
        print(f"Output shape per frame: {geom.output_shape}")
        
        if return_arrays:
            all_bmode = []
            all_ceus = []
        
        # Pre-allocate frame buffer
        frame_buffer = np.zeros(geom.output_shape, dtype=np.float32)
        
        for i in tqdm(range(n_frames), desc="Converting frames"):
            # Process B-mode (linear)
            frame_buffer.fill(0)
            self.lin_converter.convert_frame(self.linVol[i], output=frame_buffer)
            lin_vol = (frame_buffer - lin_scale['threshold']) * 255 / lin_scale['divisor']
            lin_vol = formatVolumePix(lin_vol)
            
            # Process CEUS (non-linear)
            frame_buffer.fill(0)
            self.nlin_converter.convert_frame(self.nLinVol[i], output=frame_buffer)
            nlin_vol = (frame_buffer - nlin_scale['threshold']) * 255 / nlin_scale['divisor']
            nlin_vol = formatVolumePix(nlin_vol)

            lin_vol = np.clip(lin_vol, 0, 255).astype(np.float32)
            nlin_vol = np.clip(nlin_vol, 0, 255).astype(np.float32)
            
            if save_individual:
                with open(self.destFolder / f"bmode_frame_{i}.pkl", 'wb') as f:
                    pickle.dump(lin_vol, f)
                with open(self.destFolder / f"ceus_frame_{i}.pkl", 'wb') as f:
                    pickle.dump(nlin_vol, f)
            
            if return_arrays:
                all_bmode.append(lin_vol.copy())
                all_ceus.append(nlin_vol.copy())
        
        # Save metadata
        bmodeShape = lin_vol.shape
        ceusShape = nlin_vol.shape
        
        timeconst = getattr(self.scParams, 'VDB_THREED_RT_VOLUME_RATE', 0)
        bmodeRes = [
            4.,
            bmodeDims[0] / bmodeShape[0],
            bmodeDims[1] / bmodeShape[1],
            bmodeDims[2] / bmodeShape[2],
            timeconst, 0., 0., 0.
        ]
        ceusRes = [
            4.,
            ceusDims[0] / ceusShape[0],
            ceusDims[1] / ceusShape[1],
            ceusDims[2] / ceusShape[2],
            timeconst, 0., 0., 0.
        ]
        
        with open(self.destFolder / "bmode_volume_dims.pkl", 'wb') as f:
            pickle.dump(bmodeRes, f)
        with open(self.destFolder / "ceus_volume_dims.pkl", 'wb') as f:
            pickle.dump(ceusRes, f)
        
        print(f"\nOutput saved to: {self.destFolder}")
        print(f"B-mode shape: {bmodeShape}")
        print(f"CEUS shape: {ceusShape}")
        
        if return_arrays:
            return np.array(all_bmode), np.array(all_ceus), bmodeDims, ceusDims
        else:
            return bmodeDims, ceusDims, bmodeShape, ceusShape


def main():
    # parser = argparse.ArgumentParser(
    #     description='GPU-accelerated Philips 4D ultrasound parser'
    # )
    # parser.add_argument('dataFolder', type=str, help='Parent folder of file to parse')
    # parser.add_argument('destFolder', type=str, help='Destination folder for outputs')
    # parser.add_argument('sipFilename', type=str, help='Name of SIP file to parse')
    # parser.add_argument('pixPerMm', type=float, help='Resolution of output volumes')
    # parser.add_argument('--scale', type=float, default=1.0,
    #                     help='Resolution scale factor (e.g., 4.0 for 4× resolution)')
    # parser.add_argument('--chunk-size', type=int, default=64,
    #                     help='Z-axis chunk size for GPU processing')
    # parser.add_argument('--gpu-memory', type=float, default=4.0,
    #                     help='Maximum GPU memory to use (GB)')
    # parser.add_argument('--cpu', action='store_true',
    #                     help='Force CPU mode (no GPU)')
    
    # args = parser.parse_args()
    resolution_scale = 3.0
    chunk_size_z = 64
    use_gpu = True
    gpu_memory_limit_gb = 4.0

    dataFolder = "/home/yuanshanwu/Documents/TUL/CEUS-Studies/P03/V03"
    destFolder = "/home/yuanshanwu/Documents/TUL/CEUS-Studies/P03/V03/NewInterpolation"
    sipFilename = "UCSD-P03-V03-CE1_10.37.08_mf_sip_capture_50_2_1_0.raw"
    nProcs = 2
    pixPerMm = 1.2
    
    # Create parser instance
    philips_parser = Philips4dParserGPU(
        resolution_scale,
        chunk_size_z,
        use_gpu,
        gpu_memory_limit_gb
    )
    
    # Read and prepare data
    dest_path = philips_parser.prepVolRead(
        dataFolder,
        sipFilename,
        destFolder,
        pixPerMm
    )
    
    # Process all volumes
    philips_parser.processAllVolumes(save_individual=True, return_arrays=False)
    
    print("\nDone!")


if __name__ == "__main__":
    # Example usage (comment out for command-line use)
    main()