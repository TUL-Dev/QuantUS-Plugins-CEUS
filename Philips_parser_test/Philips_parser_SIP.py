import os
import math
import pickle
from pathlib import Path
import multiprocessing as mp
from typing_extensions import Tuple, List, Iterable

import scipy
import numpy as np
from tqdm import tqdm
import argparse

class ScParams():
    def __init__(self):
        self.NUM_PLANES: int
        self.pixPerMm: float
        self.VDB_2D_ECHO_APEX_TO_SKINLINE: float
        self.VDB_2D_ECHO_START_WIDTH_GC: float
        self.VDB_2D_ECHO_STOP_WIDTH_GC: float
        self.VDB_THREED_START_ELEVATION_ACTUAL: float
        self.VDB_THREED_STOP_ELEVATION_ACTUAL: float
        self.VDB_2D_ECHO_STOP_DEPTH_SIP: float
        self.VDB_2D_ECHO_START_DEPTH_SIP: float
        self.VDB_2D_ECHO_SLACK_TIME_MM: float
        self.VDB_THREED_RT_VOLUME_RATE: float

        self.NumXmtCols: int
        self.NumRcvCols: int

class SipVolParams():
    def __init__(self):
        self.imagePitch = []
        self.numberLines = []
        self.numberAngles = []
        self.elevationIndex = []
        self.numberFocalZone = []
        self.numberLateralMultiline = []
        self.numberElevationMultiline = []

class SipVolDataStruct():
    def __init__(self):
        self.linImage: np.ndarray
        self.nLinImage: np.ndarray
        self.linVol: np.ndarray
        self.nLinVol: np.ndarray

def scanConvert3Va(rxLines, lineAngles, planeAngles, beamDist, imgSize, fovSize, z0):
    pixSizeX = 1/(imgSize[0]-1)
    pixSizeY = 1/(imgSize[1]-1)
    pixSizeZ = 1/(imgSize[2]-1)

    # Create Cartesian grid and convert to polar coordinates
    xLoc = (np.arange(0,1+(pixSizeX/2),pixSizeX)-0.5)*fovSize[0]
    yLoc = (np.arange(0,1+(pixSizeY/2),pixSizeY)-0.5)*fovSize[1]
    zLoc = np.arange(0,1+(pixSizeZ/2),pixSizeZ)*fovSize[2]
    Z, X, Y = np.meshgrid(zLoc, xLoc, yLoc, indexing='ij')

    PHI = np.arctan2(Y, Z+z0)
    TH = np.arctan2(X, np.sqrt(np.square(Y)+np.square(Z+z0)))
    R = np.sqrt(np.square(X)+np.square(Y)+np.square(Z+z0))*(1-z0/np.sqrt(np.square(Y)+np.square(Z+z0)))

    radLineAngles = np.pi*lineAngles/180
    radPlaneAngles = np.pi*planeAngles/180

    img = scipy.interpolate.interpn((beamDist, radLineAngles, radPlaneAngles), 
                                    rxLines, (R, TH, PHI), method='linear', bounds_error=False, fill_value=0)
    
    img = np.array(img)

    return img

def scanConvert3dVolumeSeries(dbEnvDatFullVolSeries, scParams, isLin=True, scale=True) -> Tuple[np.ndarray, list]:
    if len(dbEnvDatFullVolSeries.shape) != 4:
        numVolumes = 1
        nz, nx, ny = dbEnvDatFullVolSeries.shape
    else:
        numVolumes = dbEnvDatFullVolSeries.shape[0]
        nz, nx, ny = dbEnvDatFullVolSeries[0].shape
    apexDist = scParams.VDB_2D_ECHO_APEX_TO_SKINLINE # Distance of virtual apex to probe surface in mm
    azimSteerAngleStart = scParams.VDB_2D_ECHO_START_WIDTH_GC*180/np.pi # Azimuth steering angle (start) in degree
    azimSteerAngleEnd = scParams.VDB_2D_ECHO_STOP_WIDTH_GC*180/np.pi # Azimuth steering angle (end) in degree
    rxAngAz = np.linspace(azimSteerAngleStart, azimSteerAngleEnd, nx) # Steering angles in degree
    elevSteerAngleStart = scParams.VDB_THREED_START_ELEVATION_ACTUAL*180/np.pi # Elevation steering angle (start) in degree
    elevSteerAngleEnd = scParams.VDB_THREED_STOP_ELEVATION_ACTUAL*180/np.pi # Elevation steering angle (end) in degree
    rxAngEl = np.linspace(elevSteerAngleStart, elevSteerAngleEnd, ny) # Steering angles in degree
    DepthMm=scParams.VDB_2D_ECHO_STOP_DEPTH_SIP
    imgDpth = np.linspace(0, DepthMm, nz) # Axial distance in mm
    volDepth = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP *(abs(math.sin(math.radians(elevSteerAngleStart))) + abs(math.sin(math.radians(elevSteerAngleEnd)))) # Elevation (needs validation)
    volWidth = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP *(abs(math.sin(math.radians(azimSteerAngleStart))) + abs(math.sin(math.radians(azimSteerAngleEnd))))   # Lateral (needs validation)
    volHeight = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP - scParams.VDB_2D_ECHO_START_DEPTH_SIP # Axial (needs validation)
    fovSize   = [volWidth, volDepth, volHeight] # [Lateral, Elevation, Axial]
    imgSize = np.array(np.round(np.array([volWidth, volDepth, volHeight])*scParams.pixPerMm), dtype=np.uint32) # [Lateral, Elevation, Axial]

    NonLinThr=3.5e4; NonLinDiv=1.7e4
    LinThr=3e4; LinDiv=3e4
    
    # Generate image
    imgOut = []
    if numVolumes > 1:
        for k in range(numVolumes):
            rxAngsAzVec = np.linspace(rxAngAz[0],rxAngAz[-1],dbEnvDatFullVolSeries[k].shape[1])
            rxAngsElVec = np.einsum('ikj->ijk', np.linspace(rxAngEl[0],rxAngEl[-1],dbEnvDatFullVolSeries[k].shape[2]))
            curImgOut = scanConvert3Va(dbEnvDatFullVolSeries[k], rxAngsAzVec, rxAngsElVec, imgDpth,imgSize,fovSize, apexDist)
            if scale:
                if not isLin:
                    imgOut.append((curImgOut-NonLinThr)*255/NonLinDiv)
                else:
                    imgOut.append((curImgOut-LinThr)*255/LinDiv)
            else:
                curImgOut = np.array(curImgOut)
                curImgOut /= np.amax(curImgOut)
                curImgOut *= 255
                imgOut = curImgOut
                imgOut.append(curImgOut)
                
        imgOut = np.array(imgOut)
    else:
        rxAngsAzVec = np.linspace(rxAngAz[0],rxAngAz[-1],dbEnvDatFullVolSeries.shape[1])
        rxAngsElVec = np.linspace(rxAngEl[0],rxAngEl[-1],dbEnvDatFullVolSeries.shape[2])
        curImgOut = scanConvert3Va(dbEnvDatFullVolSeries, rxAngsAzVec, rxAngsElVec, imgDpth,imgSize,fovSize, apexDist)
        if scale:
            if not isLin:
                imgOut = (curImgOut-NonLinThr)*255/NonLinDiv
            else:
                imgOut = (curImgOut-LinThr)*255/LinDiv
        else:
            curImgOut = np.array(curImgOut)
            curImgOut /= np.amax(curImgOut)
            curImgOut *= 255
            imgOut = curImgOut
    
    return imgOut, fovSize

def formatVolumePix(unformattedVolume: Iterable) -> np.ndarray:
    unformattedVolume = np.array(unformattedVolume).squeeze()
    unformattedVolume = np.transpose(unformattedVolume.swapaxes(0,1))
    # unformattedVolume = np.transpose(unformattedVolume)
    
    return unformattedVolume

def readSIPscVDBParams(filename):
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
            paramAr[i] = float(paramAr[i]) # type: ignore

        if len(paramAr) == 1:
            paramValue = paramAr[0]
        else:
            paramValue = paramAr

        if (paramName == 'VDB_2D_ECHO_MATRIX_ELEVATION_NUM_TRANSMIT_PLANES'):
            scParams.NUM_PLANES = int(paramValue) # type: ignore
        elif (paramName == 'pixPerMm'):
            scParams.pixPerMm = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_APEX_TO_SKINLINE'):
            scParams.VDB_2D_ECHO_APEX_TO_SKINLINE = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_START_WIDTH_GC'):
            scParams.VDB_2D_ECHO_START_WIDTH_GC = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_STOP_WIDTH_GC'):
            scParams.VDB_2D_ECHO_STOP_WIDTH_GC = paramValue # type: ignore
        elif (paramName == 'VDB_THREED_START_ELEVATION_ACTUAL'):
            scParams.VDB_THREED_START_ELEVATION_ACTUAL = paramValue # type: ignore
        elif (paramName == 'VDB_THREED_STOP_ELEVATION_ACTUAL'):
            scParams.VDB_THREED_STOP_ELEVATION_ACTUAL = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_STOP_DEPTH_SIP'):
            scParams.VDB_2D_ECHO_STOP_DEPTH_SIP = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_START_DEPTH_SIP'):
            scParams.VDB_2D_ECHO_START_DEPTH_SIP = paramValue # type: ignore
        elif (paramName == 'VDB_2D_ECHO_SLACK_TIME_MM'):
            scParams.VDB_2D_ECHO_SLACK_TIME_MM = paramValue # type: ignore
        elif (paramName == 'VDB_THREED_RT_VOLUME_RATE'):
            scParams.VDB_THREED_RT_VOLUME_RATE = paramValue # type: ignore

    file.close()        
    print('Finished reading SIP scan converstion VDB params...')
    return scParams

def readSIP3dInterleavedV5(filename, numberOfPlanes=32, numberOfParams=5):
    print('Reading interleaved SIP volume data...')
    file = open(filename, "rb")
    param = SipVolParams()
    img = SipVolDataStruct()
    linImageList = []; nLinImageList = []
    while (True):
        tmpLine = np.fromfile(file, count=numberOfParams, dtype='<u4')
        
        if numberOfParams == 7:
            # Legacy
            param.imagePitch.append(tmpLine[0])
            param.numberLines.append(tmpLine[1])
            param.numberAngles.append(tmpLine[2])
            param.elevationIndex.append(tmpLine[3])
            param.numberFocalZone.append(tmpLine[4])
            param.numberLateralMultiline.append(tmpLine[5])
            param.numberElevationMultiline.append(tmpLine[6])
        elif numberOfParams == 5:
            # New
            param.imagePitch.append(tmpLine[0])
            param.numberLines.append(tmpLine[1])
            param.numberFocalZone.append(tmpLine[2])
            param.numberLateralMultiline.append(tmpLine[3])
            param.numberElevationMultiline.append(tmpLine[4])
        else:
            print("Unexpected number of header parameters")
            break

        # Read in enough to account for 2 frames (1 linear and 1 non-linear)
        try:
            lineBuf = np.fromfile(file, count=int(param.imagePitch[-1]/2)*param.numberLines[-1], dtype='<u2')
        except:
            break
        lineBuf = lineBuf.reshape((int(param.imagePitch[-1]/2), param.numberLines[-1]), order='F')

        linImageList.append(lineBuf[np.arange(0, lineBuf.shape[0], 2)])
        nLinImageList.append(lineBuf[np.arange(1, lineBuf.shape[0], 2)])

        if file.tell() >= os.fstat(file.fileno()).st_size:
            break

    file.close()

    img.linImage = np.array(linImageList)
    img.nLinImage = np.array(nLinImageList)
    totalNumFrames = len(img.linImage)
    totalNumFramesFullVol = totalNumFrames - (totalNumFrames % numberOfPlanes)
    numVolumes = int(totalNumFramesFullVol/numberOfPlanes)

    # Reshape data into volumes
    tmpLin = np.zeros((img.linImage[0].shape[0], img.linImage[0].shape[1], numberOfPlanes))
    tmpNLin = np.zeros((img.nLinImage[0].shape[0], img.nLinImage[0].shape[1], numberOfPlanes))
    img.linVol = np.zeros(([numVolumes] + list(tmpLin.shape)), dtype=np.uint16)
    img.nLinVol = np.zeros(([numVolumes] + list(tmpNLin.shape)), dtype=np.uint16)

    for n in range(numVolumes):
        for m in range(numberOfPlanes):
            ind = n*numberOfPlanes + m
            tmpLin[:,:,m] = img.linImage[ind]
            tmpNLin[:,:,m] = img.nLinImage[ind]
        img.linVol[n] = tmpLin
        img.nLinVol[n] = tmpNLin

    print("Finished reading interleaved SIP volume data...")
    return img

class Philips4dParser:
    def __init__(self):
        self.scParams: ScParams
        self.destFolder: Path
        self.sipVolDat: SipVolDataStruct
        
    def prepVolRead2(self, pathToData, sipFilename, destFolder, pixPerMm=1.2):
        vdbFilename = str("_".join(sipFilename.split("_")[:2]) + "_vdbDump.xml")
        scParamPath = Path(pathToData) / str(vdbFilename+"_Extras.txt")
        
        nonLinSample=2; nonLinThr=3.5e4; nonLinDiv=1.7e4
        linSample=1; linThr=3e4; linDiv=3e4
        stpSample=2

        self.scParams: ScParams = readSIPscVDBParams(scParamPath)
        if not hasattr(self.scParams, 'NUM_PLANES'):
            self.scParams.NUM_PLANES = 20
        if not hasattr(self.scParams, 'pixPerMm'):
            self.scParams.pixPerMm = pixPerMm
        if not hasattr(self.scParams, 'VDB_THREED_RT_VOLUME_RATE'):
            self.scParams.VDB_THREED_RT_VOLUME_RATE = 0
            
        numPlanes = self.scParams.NUM_PLANES
        p0 = round(numPlanes/2)
        paramLen = 5
        rfPath = Path(pathToData) / sipFilename
        params = np.fromfile(rfPath, dtype=np.int32, count=paramLen)
        numSamples = int(params[0]/stpSample)
        numLines = int(params[1])
        numPixels = numSamples * numLines
        AZ_XBR_OUT = int(params[3])
        EL_ML = int(params[4])

        buffer = np.fromfile(rfPath, dtype=np.uint16, count=-1)
        paramOffs = 2*paramLen
        numSlices = int(buffer.size / (numPixels + paramOffs))
        numVolumes = int(np.floor(numSlices / numPlanes))

        out = np.zeros((numSamples, numLines, numPlanes, numVolumes))
        for v in tqdm(range(numVolumes)):
            offs = (numPixels + paramOffs) * numPlanes * v + paramOffs
            offs = int(offs)
            for a in range(numPlanes):
                out[:,:,a,v] = buffer[offs:offs+numPixels].reshape((numSamples, numLines), order='F')
                offs += numPixels + paramOffs
            
        self.nLinVol = out[np.arange(nonLinSample-1, out.shape[0], stpSample)] # most likely contrast
        self.linVol = out[np.arange(linSample-1, out.shape[0], stpSample)] # most likely B-mode
        
        # Make dest folder
        self.destFolder = Path(destFolder)
        destFolderName = "_".join(sipFilename.split("_")[:2])
        self.destFolder = self.destFolder / Path(destFolderName)
        self.destFolder.mkdir(exist_ok=True, parents=True)

        return self.destFolder
    
    def prepVolRead(self, pathToData, sipFilename, destFolder, pixPerMm=1.2):
        # Input paths/filenames
        vdbFilename = str("_".join(sipFilename.split("_")[:2]) + "_vdbDump.xml")
        scParamFilename = str(vdbFilename+"_Extras.txt")

        self.scParams = readSIPscVDBParams(os.path.join(pathToData, scParamFilename))
        if not hasattr(self.scParams, 'NUM_PLANES'):
            self.scParams.NUM_PLANES = 20
        if not hasattr(self.scParams, 'pixPerMm'):
            self.scParams.pixPerMm = pixPerMm

        # Read in the interleaved SIP volume data time series (both linear/non-linear parts)
        sipVolDat = readSIP3dInterleavedV5(os.path.join(pathToData, sipFilename),  self.scParams.NUM_PLANES)
        self.linVol = sipVolDat.linVol
        self.nLinVol = sipVolDat.nLinVol

        # Make dest folder
        self.destFolder = Path(destFolder)
        destFolderName = "_".join(sipFilename.split("_")[:2])
        self.destFolder = self.destFolder / Path(destFolderName)
        self.destFolder.mkdir(exist_ok=True, parents=True)

        return self.destFolder

    def saveSingleVol(self, volIndices: List[int]) -> Tuple[list, list, tuple, tuple]:
        for volIndex in tqdm(volIndices):
            linVol, bmodeDims = scanConvert3dVolumeSeries(self.linVol[volIndex], self.scParams, isLin=True)
            nLinVol, ceusDims = scanConvert3dVolumeSeries(self.nLinVol[volIndex], self.scParams, isLin=False)
            bmodeDims = [bmodeDims[2], bmodeDims[0], bmodeDims[1]]
            ceusDims = [ceusDims[2], ceusDims[0], bmodeDims[1]]

            linVol = formatVolumePix(linVol)
            nLinVol = formatVolumePix(nLinVol)

            with open(self.destFolder / Path(f"bmode_frame_{volIndex}.pkl"), 'wb') as bmodeFile:
                pickle.dump(linVol, bmodeFile)
            with open(self.destFolder / Path(f"ceus_frame_{volIndex}.pkl"), 'wb') as ceusFile:
                pickle.dump(nLinVol, ceusFile)

        return bmodeDims, ceusDims, linVol.shape, nLinVol.shape
    
def sipParser(dataFolder, destFolder, sipFilename, nProcs, pixPerMm):
    procs = []
    PhilipsParser = Philips4dParser()
    volDestPath = PhilipsParser.prepVolRead(dataFolder, sipFilename, destFolder, pixPerMm)

    volInds = list(range(PhilipsParser.linVol.shape[0]))
    splitInds = np.array_split(volInds, nProcs)

    for indChunk in splitInds:
        proc = mp.Process(target=PhilipsParser.saveSingleVol, args=(indChunk,))
        procs.append(proc)

    # start processes
    for proc in procs:
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    bmodeDims, ceusDims, bmodeShape, ceusShape = PhilipsParser.saveSingleVol([0])

    timeconst = PhilipsParser.scParams.VDB_THREED_RT_VOLUME_RATE
    bmodeRes = [4., bmodeDims[0]/bmodeShape[0], bmodeDims[1]/bmodeShape[1], bmodeDims[2]/bmodeShape[2], timeconst, 0., 0., 0.]
    ceusRes = [4., ceusDims[0]/ceusShape[0], ceusDims[1]/ceusShape[1], ceusDims[2]/ceusShape[2], timeconst, 0., 0., 0.]
    with open(volDestPath / Path("bmode_volume_dims.pkl"), 'wb') as resFile:
        pickle.dump(bmodeRes, resFile)
    with open(volDestPath / Path("ceus_volume_dims.pkl"), 'wb') as resFile:
        pickle.dump(ceusRes, resFile)

if __name__ == "__main__":  # confirms that the code is under main function
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataFolder', metavar='FOLDER', type=str, nargs=1,
                        help='parent folder of file to parse')
    parser.add_argument('destFolder', metavar='DEST', type=str, nargs=1,
                        help='destination folder of outputs')
    parser.add_argument('sipFilename', metavar='FILE', type=str, nargs=1,
                        help='name of file to parse')
    parser.add_argument('nProcs', metavar='PROCS', type=int, nargs=1,
                        help='number of processes for parsing')
    parser.add_argument('pixPerMm', metavar='RES', type=float, nargs=1,
                        help='resolution of output volumes')


    args = parser.parse_args()

    dataFolder = args.dataFolder[0] #"/Users/davidspector/Downloads/wei_test"
    destFolder = args.destFolder[0]
    sipFilename = args.sipFilename[0] #"SHC-PTEST001-V02-CE01_16.05.52_mf_sip_capture_50_2_1_0.raw"
    nProcs = args.nProcs[0] #4
    pixPerMm = args.pixPerMm[0] #1.2
    procs = []
    Example = Philips4dParser()
    volDestPath = Example.prepVolRead(dataFolder, sipFilename, destFolder, pixPerMm)

    volInds = list(range(Example.linVol.shape[0]))
    splitInds = np.array_split(volInds, nProcs)

    for indChunk in splitInds:
        proc = mp.Process(target=Example.saveSingleVol, args=(indChunk,))
        procs.append(proc)

    # start processes
    for proc in procs:
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()

    bmodeDims, ceusDims, bmodeShape, ceusShape = Example.saveSingleVol([0])

    timeconst = 0 # no timeconst for now
    bmodeRes = [4., bmodeDims[0]/bmodeShape[0], bmodeDims[1]/bmodeShape[1], bmodeDims[2]/bmodeShape[2], timeconst, 0., 0., 0.]
    ceusRes = [4., ceusDims[0]/ceusShape[0], ceusDims[1]/ceusShape[1], ceusDims[2]/ceusShape[2], timeconst, 0., 0., 0.]
    with open(volDestPath / Path("bmode_volume_dims.pkl"), 'wb') as resFile:
        pickle.dump(bmodeRes, resFile)
    with open(volDestPath / Path("ceus_volume_dims.pkl"), 'wb') as resFile:
        pickle.dump(ceusRes, resFile)

# py_sipVolDat = main("/Volumes/CREST Data/", "/Volumes/CREST Data", "TJU-P001-V01-CEUS-2_10.05.43_mf_sip_capture_50_2_1_0.raw", 4, 1.2)
# To run the code 
# python Philips_parser_test/Philips_parser_SIP.py "/Volumes/T5 EVO/UCSD_3DCEUS/SIP/2025.07.23_P05V01" "/Volumes/T5 EVO/UCSD_3DCEUS/SIP/2025.07.23_P05V01/NewInterpolation" "UCSD-P05-V01-CE1_11.24.01_mf_sip_capture_50_2_1_0.raw" 4 1.2