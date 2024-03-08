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
        

        pass

    def load_model(self):
        sam = sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth")
        sam.to(device = "cuda")
        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator
            
    def bounding_boxes():
        """
        bounding box json include:
        {img: "full img path"
        bounding_box: "[x,y,width,length]"}
        """
        pass
    def segmentation_mask():
        """
        write img to source file shrunk to bouding box
        segmentation mask json include:
        {img: "bounding box img path"
        mask location: [boolean array size of img]"}
        """
        pass

    def run(self):
        mask_generator = self.load_model()
        root_path = self.source_directory
        for img_path in os.listdir(root_path):
            assert img_path.split(".")[-1] == "jpg", "File is not JPG"

            img = Image.open(os.path.join(root_path,img_path))
            data = asarray(img)
            masks = mask_generator.generate(data)

            """
            TO DO: using masks - call bounding_boxes and segemention_mask to get write metadata and image to dest location
            """


