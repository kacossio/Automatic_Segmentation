import torch
from torch.utils.data import DataLoader, Dataset
import yaml
from enum import Enum, auto
from typing import Type
import json
import pandas as pd
import os
import cv2 
import numpy as np
from torchvision import transforms
import util.augmentations as augmentations
from tqdm import tqdm
import random


class loaderType(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


class cocoLoader(Dataset):
    def __init__ (
            self, 
            config : str, 
            loaderType : Type[loaderType], 
            transform : bool = False,
            feature_extractor = None,
            category_id_subset : list = None) : 
        """
        :param config: path of config.yaml file
        :param loaderType: Instance of loaderType to define what type of loader this object will be
        :param transform: tranforms boolean for resize. Use during training only
        """

        with open(config) as f:
            self.config = yaml.safe_load(f)
        self.type = loaderType
        self.transform = transform
        self.img_path = None
        self.label_path = None
        self.img_shape = self.config["dataset"]["img_shape"]
        self.feature_extractor = feature_extractor
        self.category_id_subset = category_id_subset
        self._post_init_()

    def _post_init_(self):
        self._get_loader_type_()
        self._get_metadata_()


    def _get_loader_type_(self):
            if self.type == loaderType.TRAIN:
                self.img_path = self.config["dataset"]["train_img"]
                label_path =  self.config["dataset"]["train_annotation"]
                with open(label_path)  as f:
                    self.label = json.load(f)
            elif self.type == loaderType.VAL:
                self.img_path = self.config["dataset"]["val_img"]
                label_path =  self.config["dataset"]["val_annotation"]
                with open(label_path)  as f:
                    self.label = json.load(f)
            else:
                self.img_path = self.config["dataset"]["test_img"]
                self.label = None


    def _get_metadata_(self):
        print(f"Loading {self.type} metadata")
        metadata = {}
        real_indx = 0
        for indx in tqdm(range(self.__len__())):
            #image_metadata = [i for i in self.label["images"] if i['id'] == self.label["annotations"][indx]["image_id"]][0]
            if self.category_id_subset:
                if self.label["annotations"][indx]["category_id"] not in self.category_id_subset:
                    continue
            x = {}
            x["bbox"] = (self.label["annotations"][indx]["bbox"])
            x["category_id"] = (self.label["annotations"][indx]["category_id"])
            x["image_id"] = (self.label["annotations"][indx]["image_id"])
            id = x["image_id"]
            x["file_name"] = (f"{id:012d}.jpg")
            x["raw_annotation"] = [self.label["annotations"][indx]]
            metadata[real_indx]=x
            real_indx += 1
        self.metadata = metadata


    def __getitem__(self, idx) : # function to return dataset size
        img_path = os.path.join(self.img_path,self.metadata[idx]['file_name'])
        img = np.array(cv2.imread(img_path))
        labels = self.metadata[idx]["category_id"]
        boxes = self.metadata[idx]["bbox"]
        if self.transform:
            boxes = augmentations.resize_img_bbox(img.shape,boxes,self.img_shape)
            transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(self.img_shape)])
            img = transform(img)
        if self.feature_extractor:
            annot = {"image_id" : self.metadata[idx]["image_id"], "annotations" :self.metadata[idx]["raw_annotation"] }
            encoding = self.feature_extractor(images=img, annotations=annot, return_tensors="pt")
            pixel_value = np.array(encoding["pixel_values"].squeeze()) # remove batch dimension
            target = encoding["labels"][0] # remove batch dimension
            return pixel_value,target

        boxes = torch.FloatTensor(boxes) 
        return img, boxes, labels

    def __len__(self): 
        try:
            return len(self.metadata)
        except:
            return len(self.label["images"])
    
    def balance_dataset(self,max_count):
        metadata = {}
        real_indx = 0
        for cls in tqdm(self.category_id_subset):
            counter = 0
            for _,key in enumerate(self.metadata):
                if self.metadata[key]["category_id"] == cls:
                    metadata[real_indx] = self.metadata[key]
                    real_indx += 1
                    counter += 1
                    if counter >= max_count:
                        break
        self.metadata = metadata
        

    """
    To do: resize img and bbox to (224,224)
    """



