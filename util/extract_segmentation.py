"""
Used to extract bounding boxes and segmentation masks using Segement Anything
"""

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt
import numpy as np
import torch
from pathlib import Path
import yaml
import os
import glob
import json
import pickle

class Segmenter():

    def __init__(
        self,
        config : Path):

        with open(config) as f:
            self.config = yaml.safe_load(f)

        self.source_directory = self.config["source_directory"]
        self.dest_directory = self.config["dest_directory"]
        self.model_weight = self.config["model_weights"]
        self.cache_file = self.config["cache_file"]
        self.cache = {}
        with open(self.cache_file, 'rb') as openfile:
            self.cache = pickle.load(openfile)
        self.metadata = {}
        pass

    def _update_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache,f)

    def load_model(self):
        sam = sam_model_registry["vit_h"](checkpoint=self.model_weight)
        try:
            sam.to(device = "cuda")
        except:
            print("no cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator
            
    def bounding_boxes(self,mask,img):
        x1 = mask['bbox'][0]
        y1 = mask['bbox'][1]
        width = mask['bbox'][0] + mask['bbox'][2]
        length = mask['bbox'][1] + mask['bbox'][3]        
        cropped_img = img[y1:length,x1:width]
        return cropped_img
    
    def segmentation_mask(self,mask,data):
        """
        write img to source file shrunk to bouding box
        segmentation mask json include:
        {img: "bounding box img path"
        mask location: [boolean array size of img]"}
        """
        pass
    
    def get_masks(self,mask_generator,img_path,cached = True):
        img = Image.open(img_path)
        data = asarray(img)
        if cached:
            if img_path not in self.cache:
                masks = mask_generator.generate(data)
                self.cache[img_path] = masks
            else:
                masks = self.cache[img_path]
        else:
            masks = mask_generator.generate(data)
        return data, masks

    def run(self):
        mask_generator = self.load_model()
        root_path = self.source_directory
        for img_path in glob.glob(root_path + '/*.jpg'):
            assert img_path.split(".")[-1] == "jpg", "File is not JPG"
            data, masks = self.get_masks(mask_generator,img_path)
            for num, mask in enumerate(masks):
                self.img_metadata = {}
                self.img_metadata["bbox"] = mask["bbox"]
                self.img_metadata["segmentation"] = mask["segmentation"]
                cropped_img = self.bounding_boxes(mask,data)
                im = Image.fromarray(cropped_img)
                im.save(os.path.join(self.dest_directory,f'cropped_img_{os.path.basename(img_path).split(".")[0]}_{num}.jpg'))
                self.segmentation_mask(mask,data)
        self._update_cache()

    """
    TO DO: using masks - call bounding_boxes and segemention_mask to get write metadata and image to dest location
            Initialize cache file on very first run
            Better persisent cacheing of masks
    """


