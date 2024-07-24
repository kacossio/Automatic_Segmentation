"""
Used to extract bounding boxes and segmentation masks using Segement Anything
"""

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
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
        if "cache_file" in self.config.keys():
            self.cache_flag = True
            self.cache_file = self.config["cache_file"]
        else:
            self.cache_flag = False
        self.cache = {}
        if self.cache_flag:
            with open(self.cache_file, 'rb') as openfile:
                self.cache = pickle.load(openfile)
        self.metadata = {}

    def _update_cache(self):
        if self.cache_flag:
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
        self.x1 = mask['bbox'][0]
        self.y1 = mask['bbox'][1]
        self.length = self.x1 + mask['bbox'][2]
        self.height = self.y1 + mask['bbox'][3]        
        cropped_img = img[self.y1:self.height,self.x1:self.length]
        return cropped_img

    def _annotation_init(self,filename,data):
        self.annotation = {}
        self.annotation["images"] = {
                                        "file_name": filename,
                                        "height": data.shape[0],
                                        "width": data.shape[1],
                                    }
        self.annotation_list = []

    def _write_annotation(self,cropped_img_path):
        annoation = {
                        "filename": cropped_img_path,
                        "segmentation" : self.img_metadata["segmentation"][self.y1:self.height,self.x1:self.length].tolist(),
                        "bbox" : self.img_metadata["bbox"]
                    }
        self.annotation_list.append(annoation)

    def _save_annotation(self,img_path):
        self.annotation["annotation"] = self.annotation_list
        json_filename = os.path.join(self.dest_directory,f'{os.path.basename(self.annotation["images"]["file_name"]).split(".")[0]}_.json')
        with open(json_filename, "w") as outfile:
            json_object = json.dumps(self.annotation)
            outfile.write(json_object)
    
    def get_masks(self,mask_generator,img_path):
        img = Image.open(img_path)
        data = np.asarray(img)
        if self.cache_flag:  
            if img_path not in self.cache:
                masks = mask_generator.generate(data)
                self.cache[img_path] = masks
            else:
                masks = self.cache[img_path]
        else:
            masks = mask_generator.generate(data)
        self._annotation_init(img_path,data)
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
                cropped_img_path = os.path.join(self.dest_directory,f'cropped_img_{os.path.basename(img_path).split(".")[0]}_{num}.jpg')
                im.save(cropped_img_path)
                self._write_annotation(cropped_img_path)
            self._save_annotation(cropped_img_path)
        self._update_cache()



