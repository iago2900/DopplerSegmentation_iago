from typing import Tuple, Dict
import cv2
import torch
import torch.utils
import torch.utils.data
import numpy as np
import dill
import sak
import sak.torch

from sak.__ops import required
from sak.__ops import check_required


class DatasetDopplerDict(torch.utils.data.Dataset):
    def __init__(self, inputs: dict, **kwargs):
        # Divide inputs
        x     = inputs.get("x", required)
        label = inputs.get("label", required)
        y_1d  = inputs.get("y_1d", required)
        y_2d  = inputs.get("y_2d", required)
        shape = inputs.get("shape", (256,512))
        dtype = inputs.get("dtype", 'float32')

        # Check inputs
        check_required(self,{"x": x, "label": label, "y_1d": y_1d, "y_2d": y_2d})
        assert (set(x) == set(label)) and (set(x) == set(y_1d)) and (set(x) == set(y_2d))

        # Store in class
        self.x = x
        self.label = label
        self.y_1d = y_1d
        self.y_2d = y_2d
        self.dtype = dtype
        self.len = len(self.x)
        self.shape = tuple(shape)

        # Get order of keys
        self.keys = list(self.x)
        
        # Convert dtypes
        for k in x:
            self.x[k] = self.x[k].astype(self.dtype)
            self.y_1d[k] = self.y_1d[k].astype(self.dtype)
            self.y_2d[k] = self.y_2d[k].astype(self.dtype)
        
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        if i == self.len:
            raise StopIteration
            
        # Get key
        k     = self.keys[i]
        x     = np.copy(self.x[k])
        label = np.copy(self.label[k])
        y_1d  = np.copy(self.y_1d[k])
        y_2d  = np.copy(self.y_2d[k])
        
        # Get shape
        shape_y,shape_x = x.shape
        
        # Crop unevenly (x dimension)
        if x.shape[1] != self.shape[1]:
            onset  = np.random.randint(x.shape[1]-self.shape[1])
            offset = (x.shape[1]-self.shape[1])-onset
            x      = x[:,onset:-offset]
            y_1d   = y_1d[onset:-offset]
            y_2d   = y_2d[:,onset:-offset]
        
        # Reshape to 256x512 elements
        x    = cv2.resize(x,self.shape[::-1])[None,]
        y_1d = cv2.resize(y_1d[:,None],(1,self.shape[-1]))[None,:,0]*(self.shape[0]/shape_y)
        y_2d = cv2.resize(y_2d,self.shape[::-1])[None,]
        
        return {"x": x, "label": label, "y_1d": y_1d, "y_2d": y_2d}


