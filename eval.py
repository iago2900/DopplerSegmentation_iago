from argparse import ArgumentParser

import os
import os.path
import copy
import shutil
import datetime
import pathlib
import random
import math
import json
from cv2 import PCAProject
import tqdm
import time
import pydicom
import torch
import numpy as np
import scipy as sp
import sak
import sak.data
import sak.torch
import sklearn
import sklearn.preprocessing
import sak.torch.train
import pickle
import src.load_data
import train_doppler
import matplotlib.pyplot as plt
import torch
import sak
import pickle
import cv2
import sak.torch.image.data.augmentation
import src.load_data


def main(config_file: str):

    # escoger modelo:
    model_n = 'Unet_5levels'

    # 1) Load configuration

    with open(config_file, "r") as f:
        config = json.load(f)

    # 2) Load test set
    with open(os.path.join(config["basedir"], "TrainedModels", model_n, "execution_info.pkl"), "rb") as f:
        keys_test = pickle.load(f)["keys_test"]

    # 3) Load model
    model = torch.load(os.path.join(config['savedir'], model_n, "model_best.model"))
    model.eval()

    # 4) Load images
    data = [os.path.join(config["datadir"],i) for i in keys_test]

    path = data[0]
    dicom = pydicom.dcmread(path)
    x0,x1,y0,y1,xref,yref,deltax,deltay,unitsx,unitsy = src.load_data.get_frame(dicom,"doppler")
    x = dicom.pixel_array.copy()[y0:y1,x0:x1,]
    if dicom.get("PhotometricInterpretation", "RGB") in ['YBR_FULL', 'YBR_FULL_422']:
        x = src.load_data.convert_ybr_to_rgb(x)
    x = x[...,0] # to grayscale

    # 5) Interpolation & cropping
    shape = (256,512) # quiero que sea 256x512
    prop = shape[1] // shape[0]
    x_side = x.shape[0]*prop
    x = x[:,0:x_side]
    x = cv2.resize(x,(512,256))[None,]
    x = x.squeeze()

    # 6) To tensor
    x_torch = torch.tensor(x)[None,None]
    inputs = {'x' : x_torch.float()}
    sak.torch.image.data.augmentation.RescaleIntensity(inputs)

    # 7) Generate output
    output = model(inputs)
    out_numpy = output['sigmoid'].cpu().detach().numpy().squeeze()

    plt.imshow(out_numpy)

    """
    Comprobar si la red está generalizando bien. Para ello, haría un plot de la imagen original con la segmentación superpuesta. 
    Además, compararía la predicción con el GT (puedes usar dos métricas por ahora, el RMSE y el Dice Score, 
    aunque tenemos que mirar mejores métricas para comprobar los resultados). Para ello, de cada columna de la imagen 
    tienes que ver la localización del píxel central de la máscara predicha.
    """




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file",    type=str, required=True,  help="location of config file (.json)")
    parser.add_argument("--hpc",            type=bool, default=False, help="mark if executed in HPC")
    args = parser.parse_args()

    main(args.config_file)
