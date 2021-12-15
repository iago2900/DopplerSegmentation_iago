from typing import Callable, Dict, Tuple, List, Union

import os
import json
import glob
import math
import tqdm
import numpy as np
import scipy as sp
import skimage
import skimage.util
import sak
import cv2
import pydicom
import src.splines

from scipy.ndimage import gaussian_filter1d


def get_region(dicom: pydicom.Dataset, ix_region: int) -> Tuple[int,int,int,int,int,int,float,float,int,int]:
    region  = dicom.get("SequenceOfUltrasoundRegions", [])[ix_region]
    x0      = region.RegionLocationMinX0
    x1      = region.RegionLocationMaxX1
    y0      = region.RegionLocationMinY0
    y1      = region.RegionLocationMaxY1
    xref    = region.ReferencePixelY0 if hasattr(region,"ReferencePixelX0")        else None
    yref    = region.ReferencePixelY0 if hasattr(region,"ReferencePixelY0")        else None
    deltax  = region.PhysicalDeltaX   if hasattr(region,"PhysicalDeltaX")          else None
    deltay  = region.PhysicalDeltaY   if hasattr(region,"PhysicalDeltaY")          else None
    unitsx  = match_units(region.PhysicalUnitsXDirection) if hasattr(region,"PhysicalUnitsXDirection") else None
    unitsy  = match_units(region.PhysicalUnitsYDirection) if hasattr(region,"PhysicalUnitsYDirection") else None

    return x0,x1,y0,y1,xref,yref,deltax,deltay,unitsx,unitsy


def get_frame(dicom: pydicom.Dataset, type: str = "doppler") -> Tuple[int,int,int,int,int,int,float,float,int,int]:
    # Assert types
    if   type.lower() == "doppler":
        allowedSpatialFormats = [3]
        allowedDataTypes      = [3,4]
    elif type.lower() == "bmode":
        allowedSpatialFormats = [1]
        allowedDataTypes      = [1,2]
    elif type.lower() == "mmode":
        allowedSpatialFormats = [2]
        allowedDataTypes      = [1]
    else:
        raise ValueError(f"Frame type '{type}' not in ['doppler', 'bmode', 'mmode']")

    # Initialize values
    x0,x1,y0,y1,xref,yref,deltax,deltay,unitsx,unitsy = (None,None,None,None,None,None,None,None,None,None,)

    # Aggregate same doppler regions (DICOM format)
    first_match = False
    all_regions = dicom.get("SequenceOfUltrasoundRegions", [])
    for i,region in enumerate(all_regions):
        if ("RegionDataType" not in region):
            x0,x1,y0,y1,xref,yref,deltax,deltay,unitsx,unitsy = get_region(dicom,0)
        else:
            if (region.RegionDataType in allowedDataTypes) and (region.RegionSpatialFormat in allowedSpatialFormats):
                if first_match == False:
                    x0,x1,y0,y1,xref,yref,deltax,deltay,unitsx,unitsy = get_region(dicom,i)
                    first_match = True
                
                # Check if next region can be aggregated
                x0_new,x1_new,y0_new,y1_new,_,_,_,_,_,_ = get_region(dicom,i)
                if (y0 == y0_new) and (y1 == y1_new):
                    if x0_new <= x0:
                        x0 = x0_new
                    if x1_new >= x1:
                        x1 = x1_new
    
    return x0,x1,y0,y1,xref,yref,deltax,deltay,unitsx,unitsy


def get_modality(dicom: pydicom.Dataset) -> Union[str,None]:
    if "SequenceOfUltrasoundRegions" not in dicom:
        return None

    spatialformats,datatypes = [],[]
    for region in dicom.SequenceOfUltrasoundRegions:
        if ("RegionSpatialFormat" not in region) or ("RegionDataType" not in region):
            return None
        spatialformats.append(region.RegionSpatialFormat)
        datatypes.append(region.RegionDataType)
    spatialformats,datatypes = np.array(spatialformats),np.array(datatypes)

    # Determine type of image for analysis
    if   np.any(spatialformats == 2): # M-Mode
        modality = "M-Mode"
    elif np.any(spatialformats == 3):
        ix = np.where(spatialformats == 3)[0][0]
        if   datatypes[ix] == 3: # PW Spectral Doppler
            modality = "PW Doppler" # NEEDED DISTINGUISHING PW AND TDI (!!!)
        elif datatypes[ix] == 4: # CW Spectral Doppler
            modality = "CW Doppler"
        else: # Unknown
            modality = None
    elif np.all(spatialformats == 1) and (spatialformats.size == 1):
        modality = "2D"
    else:
        modality = None

    return modality


def match_units(unit: int) -> str:
    """https://dicom.innolitics.com/ciods/us-image/general-image/00880200/00280004"""
    if   unit == 0x0000: unit = "NA"
    elif unit == 0x0001: unit = "%"
    elif unit == 0x0002: unit = "dB"
    elif unit == 0x0003: unit = "cm"
    elif unit == 0x0004: unit = "s"
    elif unit == 0x0005: unit = "hz"
    elif unit == 0x0006: unit = "dB/s"
    elif unit == 0x0007: unit = "cm/s"
    elif unit == 0x0008: unit = "cm2"
    elif unit == 0x0009: unit = "cm2/s"
    elif unit == 0x000A: unit = "cm3"
    elif unit == 0x000B: unit = "cm3/s"

    return unit

def convert_ybr_to_rgb(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 4:
        return np.vstack([convert_ybr_to_rgb(a)[np.newaxis] for a in x])
    else:
        temp = x[..., 1].copy()
        x[..., 1] = x[..., 2]
        x[..., 2] = temp
        return cv2.cvtColor(x, cv2.COLOR_YCR_CB2RGB)

def convert_rgb_to_ybr(x_rgb: np.ndarray) -> np.ndarray:
    if len(x_rgb.shape) == 4:
        return np.vstack([convert_rgb_to_ybr(a)[np.newaxis] for a in x_rgb])
    else:
        x = cv2.cvtColor(x_rgb,cv2.COLOR_RGB2YCR_CB)
        temp = x[..., 2].copy()
        x[..., 2] = x[..., 1]
        x[..., 1] = temp
        return x

def get_curve(curve: dict, x0: int = 0):
    # Match onset/offsets
    curves_x = np.array(curve["x"])-x0
    curves_y = np.array(curve["y"])
    curves_t = np.array(curve["type"])

    #### Generate curve ####
    # Interpolate spline according to generation algorithm in the image analysis platform
    points = src.splines.getSpline(
        [{
            "x": curves_x[i], 
            "y": curves_y[i], 
            "type": curves_t[i]
        } for i in range(curves_x.size)],
        "monotone"
    )
    curves_x = np.array([pt["x"] for pt in points])
    curves_y = np.array([pt["y"] for pt in points])

    return curves_x,curves_y

def extract_region(dicom, region: str = "doppler", gray: str = None):
    # Get frame and physical units
    x0,x1,y0,y1,xref,yref,deltax,deltay,unitsx,unitsy = get_frame(dicom,region)

    # Retrieve pixel array
    x = np.copy(dicom.pixel_array)

    # Color-correct dicom, if necessary
    if dicom.get("PhotometricInterpretation", None) == 'YBR_FULL_422':
        x = convert_ybr_to_rgb(x)

    # Apply frame & discard color channels
    x = x[y0:y1,x0:x1,:]
    if gray is not None:
        if gray.lower() in ["cv","cv2","opencv"]:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        else:
            x = x[...,0]
    
    return x

