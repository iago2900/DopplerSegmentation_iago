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

import src.load_data

def main(config_file: str, hpc: bool):
    ###########################################################################
    # 1) Load configuration
    root,model_name,ext = sak.splitrfe(config_file)
    config = sak.load_config(config_file, f"ALL_{model_name}")
    check_cuda = torch.Tensor([1.,]).cuda().is_cuda

    ###########################################################################
    # 2) Load data
    # 2.1) Load curves
    curves_x = sak.load_data(os.path.join(config["datadir"],"x_coordinates.csv"))
    curves_y = sak.load_data(os.path.join(config["datadir"],"y_coordinates.csv"))

    # 2.2) Load label, group, origin database and control points
    label    = sak.load_data(os.path.join(config["datadir"],"labels.csv"),dtype=str)
    database = sak.load_data(os.path.join(config["datadir"],"databases.csv"),dtype=str)
    group    = {}
    for k in label: # Convert from array to string
        label[k]    = str(label[k][0])
        database[k] = str(database[k][0])
        group[k]    = f"{database[k]}_{label[k]}"
    with open(os.path.join(config["datadir"],"control_points.json"),"r") as f:
        control_points = json.load(f)

    # 2.3) Discard datasets if not required
    if not config.get("keep_pediatric",True):
        for k in list(curves_x):
            if database[k] in ['aduheart', 'dcmsickkids', 'hclinicbcn', 'hhasseltleuven']:
                curves_x.pop(k)
                curves_y.pop(k)
                label.pop(k)
                database.pop(k)
                group.pop(k)
                control_points.pop(k)
    if not config.get("keep_hitachi",False):
        for k in list(curves_x):
            if database[k].lower() in ['hitachi']:
                curves_x.pop(k)
                curves_y.pop(k)
                label.pop(k)
                database.pop(k)
                group.pop(k)
                control_points.pop(k)
    if not config.get("keep_fetal",True):
        for k in list(curves_x):
            if database[k] in ["Gates","tof"]:
                curves_x.pop(k)
                curves_y.pop(k)
                label.pop(k)
                database.pop(k)
                group.pop(k)
                control_points.pop(k)
    if not config.get("keep_vessels",True):
        for k in list(curves_x):
            if ("valve" not in label[k].lower()) and ("tdi" not in label[k].lower()):
                curves_x.pop(k)
                curves_y.pop(k)
                label.pop(k)
                database.pop(k)
                group.pop(k)
                control_points.pop(k)
    if not config.get("keep_valves",True):
        for k in list(curves_x):
            if ("valve" in label[k].lower()):
                curves_x.pop(k)
                curves_y.pop(k)
                label.pop(k)
                database.pop(k)
                group.pop(k)
                control_points.pop(k)
    if not config.get("keep_tdi",True):
        for k in list(curves_x):
            if ("tdi" in label[k].lower()):
                curves_x.pop(k)
                curves_y.pop(k)
                label.pop(k)
                database.pop(k)
                group.pop(k)
                control_points.pop(k)
            
        
    # 2.4) Get list of keys
    list_keys = list(curves_x)

    ###########################################################################
    # 3) Convert data to mask
    # 3.1) Ground truth generation parameters
    min_size_x = 512
    window = 4
    sigma = 1.0

    # 3.2) Instantiate outputs
    inputs = {
        "x"      : {},
        "y_1d"   : {},
        "y_2d"   : {},
        "label"  : {},
        "group"  : {},
    }

    # 3.3) Load data

    line_width = 20

    counter_skipped = 0
    print("Loading data...")
    time.sleep(0.5) # Otherwise the print messes up the tqdm thingy
    iterator = sak.get_tqdm(list_keys,config.get("iterator", ""))
    for ix_key,k in enumerate(iterator):
        # Get dicom
        dicom = pydicom.dcmread(os.path.join(config["datadir"],k))
        
        # Get pixel array
        x0,x1,y0,y1,xref,yref,deltax,deltay,unitsx,unitsy = src.load_data.get_frame(dicom,"doppler")
        doppler = dicom.pixel_array.copy()[y0-line_width:y1+line_width,x0-line_width:x1+line_width,]
        if dicom.get("PhotometricInterpretation", None) == 'YBR_FULL_422':
            doppler = src.load_data.convert_ybr_to_rgb(doppler)
        
        # Convert to grayscale
        # doppler = cv2.cvtColor(doppler, cv2.COLOR_BGR2GRAY)
        doppler = doppler[...,0]

        # Get curves
        envelope_x = np.array(curves_x[k])-x0
        envelope_y = np.array(curves_y[k])
        
        # Some ground truths are out of bounds - skip these, they do not contain a full cardiac cycle
        filter_bounds = (envelope_x < 0) | (envelope_x >= doppler.shape[1]) 
        if filter_bounds.sum() != 0:
            counter_skipped += 1
            continue
        
        ####################### CHOOSING A REPRESENTATION #######################
        # Boolean mask from the reference line to the curve position            #
        mask = np.zeros(doppler.shape[:2],dtype=bool)

        gt_x_full = envelope_x.copy()
        gt_y_full = envelope_y.copy()
        for i in range(envelope_x.shape[0]-1):
            if envelope_y[i+1]>envelope_y[i]:
                s = 1
            else:
                s = -1
            for y in range(envelope_y[i]+1,envelope_y[i+1],s):
                gt_x_full = np.insert(gt_x_full,i+1,envelope_x[i])
                gt_y_full = np.insert(gt_y_full,i+1,y)
        for i in range(-line_width//2,line_width//2):
            for j in range(-line_width//2,line_width//2):
                mask[gt_y_full+i,gt_x_full+j] = 1
        
        #                                                                       #
        ####################### CHOOSING A REPRESENTATION #######################

        # Crop image and ground truth so that it only represents GT span
        doppler_cropped         = doppler[:,envelope_x]
        mask_cropped            =    mask[:,envelope_x]

        # Repeat alongside x axis (2nd dimension to have at least min_size_x size)
        try:
            repetitions      = math.ceil(min_size_x/doppler_cropped.shape[1])
            envelope_y_tiled = np.concatenate(     [envelope_y]*repetitions)
            mask_tiled       = np.concatenate(   [mask_cropped]*repetitions,axis=1)
            doppler_tiled    = np.concatenate([doppler_cropped]*repetitions,axis=1)
        except ZeroDivisionError:
            continue

        # Apply gaussian filter to smooth borders of tiling
        size_fundamental = doppler_tiled.shape[1]//repetitions
        locations = [(i+1)*size_fundamental-1 for i in range(repetitions-1)]
        for loc in locations:
            onset = loc-window+1
            offset = loc+window
            doppler_tiled[:,onset:offset] = sp.ndimage.gaussian_filter1d(
                doppler_tiled[:,onset:offset],sigma,axis=1
            )

        # Store information for dataset
        inputs["x"][k]     = doppler_tiled
        inputs["y_1d"][k]  = envelope_y
        inputs["y_2d"][k]  = mask_tiled
        inputs["label"][k] = label[k]
        inputs["group"][k] = label[k] # group[k]

    print(f" └-> Done! Skipped {counter_skipped} files due to ground truth extending beyond image bounds")
    print(f"\nEncoding labels...")

    # 3.2) Encode labels
    encoder = sklearn.preprocessing.LabelEncoder()
    encoder.fit(np.unique(list(inputs["label"].values())))
    for k in inputs["x"]:
        inputs["label"][k] = encoder.transform([inputs["label"][k]]).squeeze()
    print(f" └-> Done!")

    ###########################################################################
    # 4) Generate dataloaders
    # 4.1) Split data into train, validation and test sets
    print(f"\nCreating dataloaders...")
    keys_train,keys_valid,keys_test = None,None,None
    nested_input = list(sak.find_nested(config["model"],"class", "torch.load"))
    nested_input = nested_input[0] if len(nested_input) != 0 else {"class": ""}
    if nested_input["class"] == "torch.load":
        model_path = nested_input["arguments"]["f"]
        info_path = os.path.join(os.path.split(model_path)[0],"execution_info.pkl")

        if os.path.isfile(info_path):
            exec_info = sak.pickleload(info_path)

            # Retrieve keys
            if ("keys_train" in exec_info) and ("keys_valid" in exec_info) and ("keys_test" in exec_info):
                keys_train = [k for k in exec_info["keys_train"] if k in list_keys]
                keys_valid = [k for k in exec_info["keys_valid"] if k in list_keys]
                keys_test  = [k for k in exec_info["keys_test" ] if k in list_keys]

                inputs_train = {k1: {k2: inputs[k1][k2] for k2 in inputs[k1] if k2 in keys_train} for k1 in inputs}
                inputs_valid = {k1: {k2: inputs[k1][k2] for k2 in inputs[k1] if k2 in keys_valid} for k1 in inputs}
                inputs_test  = {k1: {k2: inputs[k1][k2] for k2 in inputs[k1] if k2 in keys_test } for k1 in inputs}

                keys_train = keys_train if len(keys_train) != 0 else None
                keys_valid = keys_valid if len(keys_valid) != 0 else None
                keys_test  = keys_test  if len(keys_test ) != 0 else None

                if (keys_train is not None) and (keys_valid is not None) and (keys_test is not None):
                    print(f" ├-> Dividing into groups with keys: {encoder.classes_}!")
    if (keys_train is None) or (keys_valid is None) or (keys_test is None):
        sak.SeedSetter(123456)
        splitter = sak.data.SplitterTrainValidTest()
        inputs_train,inputs_valid,inputs_test = splitter(inputs)

    # 4.2) Get datasets
    dataset_info_train = config["dataset"].copy()
    dataset_info_valid = config["dataset"].copy()
    dataset_info_test  = config["dataset"].copy()
    dataset_info_train["arguments"] = {**dataset_info_train.get("arguments",{}), "inputs": inputs_train}
    dataset_info_valid["arguments"] = {**dataset_info_valid.get("arguments",{}), "inputs": inputs_valid}
    dataset_info_test["arguments"]  = { **dataset_info_test.get("arguments",{}), "inputs":  inputs_test}
    dataset_train = sak.from_dict(dataset_info_train)
    dataset_valid = sak.from_dict(dataset_info_valid)
    dataset_test  = sak.from_dict(dataset_info_test)
    del dataset_info_train,dataset_info_valid,dataset_info_test # Free up memory

    # 4.3) Get dataloaders
    loader_info_train = config["dataloader"].copy()
    loader_info_valid = config["dataloader"].copy()
    loader_info_test  = config["dataloader"].copy()

    loader_info_train["arguments"] = {**loader_info_train.get("arguments",{}), "dataset": dataset_train}
    loader_info_valid["arguments"] = {**loader_info_valid.get("arguments",{}), "dataset": dataset_valid}
    loader_info_test["arguments"]  = { **loader_info_test.get("arguments",{}), "dataset":  dataset_test}

    loader_train = sak.from_dict(loader_info_train)
    loader_valid = sak.from_dict(loader_info_valid)
    loader_test  = sak.from_dict(loader_info_test)
    print(f" └-> Done!")

    ###########################################################################
    # 5) Define model and execution state
    print(f"\nCreating model...")
    model = sak.from_dict(config["model"]).train()
    print(f" └-> Done!")

    print(f"\nGenerating training state...")
    config["optimizer"]["arguments"] = {**config["optimizer"].get("arguments",{}), "params": model.parameters()}
    state = {
        'epoch'         : 0,
        'device'        : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'optimizer'     : sak.from_dict(config["optimizer"]),
        'root_dir'      : './'
    }
    if "scheduler" in config:
        config["scheduler"]["arguments"] = {**config["scheduler"].get("arguments",{}), "optimizer": state["optimizer"]}
        state["scheduler"] = sak.from_dict(config["scheduler"])
    print(f" └-> Done!")

    ###########################################################################
    # 5) Save execution information
    # 5.1) Save model-generating file
    print(f"\nCopying files to keep track...")
    sak.pickledump({"sak.version": sak.__version__,
                    "encoder":     encoder,
                    "keys_train":  dataset_train.keys,
                    "keys_valid":  dataset_valid.keys,
                    "keys_test":   dataset_test.keys,}, os.path.join(config["savedir"],"execution_info.pkl"))
    shutil.copyfile(config_file,os.path.join(config["savedir"],os.path.split(config_file)[1]))
    for i,file in enumerate(config.get("saved_files",[]) + [os.path.abspath(__file__)]):
        split_file = os.path.realpath(file).split(os.sep)
        if split_file[0] == "":
            split_file = split_file[1:]
        fname = "_".join(split_file)
        try:
            shutil.copyfile(file,os.path.join(config["savedir"],fname))
        except FileNotFoundError:
            continue
    print(f" └-> Done!")

    ###########################################################################
    # 6) Train model
    print(f"\nTraining model (will take a while)...")
    time.sleep(0.5)
    sak.torch.train.train_valid_model(model,state,config,loader_train,loader_valid)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file",    type=str, required=True,  help="location of config file (.json)")
    parser.add_argument("--hpc",            type=bool, default=False, help="mark if executed in HPC")
    args = parser.parse_args()

    main(args.config_file, args.hpc)



