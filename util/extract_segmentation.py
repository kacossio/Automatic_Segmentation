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

class Segmenter():

    def __init__(
        self,
        config : Path):

        with open(config) as f:
            self.config = yaml.safe_load(f)

        self.source_directory = self.config["source_directory"]
        self.dest_directory = self.config["dest_directory"]
        self.model_weight = self.config["model_weights"]
        self.metadata = {}
        

        pass

    def load_model(self):
        sam = sam_model_registry["vit_h"](checkpoint=self.model_weight)
        sam.to(device = "cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator
            
    def bounding_boxes(self,mask,img):
        x1 = mask['bbox'][0]
        y1 = mask['bbox'][1]
        width = mask['bbox'][0] + mask['bbox'][2]
        length = mask['bbox'][1] + mask['bbox'][3]        
        cropped_img = img[y1:length,x1:width]
        return cropped_img
    
    def segmentation_mask(self):
        """
        write img to source file shrunk to bouding box
        segmentation mask json include:
        {img: "bounding box img path"
        mask location: [boolean array size of img]"}
        """
        return 0 

    def run(self):
        mask_generator = self.load_model()
        root_path = self.source_directory
        for img_path in os.listdir(root_path):
            assert img_path.split(".")[-1] == "jpg", "File is not JPG"
            img = Image.open(os.path.join(root_path,img_path))
            data = asarray(img)
            masks = mask_generator.generate(data)
            for mask in masks:
                self.img_metadata = {}
                self.img_metadata["bbox"] = mask["bbox"]
                self.img_metadata["segmentation"] = mask["segmentation"]
                cropped_img = self.bounding_boxes(mask,img)
                im = Image.fromarray(cropped_img)
                im.save(os.path.join(self.dest_directory,("cropped_img_"+str(img_path))))
                self.segmentation_mask(mask,img)

            """
            TO DO: using masks - call bounding_boxes and segemention_mask to get write metadata and image to dest location
            """


