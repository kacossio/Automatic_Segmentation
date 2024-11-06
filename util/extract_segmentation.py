"""
Used to extract bounding boxes and segmentation masks using Segement Anything
"""
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import yaml
import os
import glob
import json
import pickle
from tqdm import tqdm
from typing import Type, List, Dict, Tuple

class Segmenter():

    def __init__(self, config: Path):
        """
        This class creates the segmenter object which runs on images in a folder based on a yaml file

        Arguments:
        config: Yaml file with path to source directory, destination directory, model weights, and cache file
        """

        with open(config) as f:
            self.config = yaml.safe_load(f)

        self.source_directory = self.config["source_directory"]
        self.dest_directory = self.config["dest_directory"]
        self.model_weights = self.config["model_weights"]
        self.model_config = self.config["model_config"]
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")
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
        """
        Updates cache file if specified in config.yaml
        """

        if self.cache_flag:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache,f)

    def load_model(self) -> Type[SAM2AutomaticMaskGenerator]:
        """
        Loads sam model using defined weights from config.yaml

        Returns:
            Type[SamAutomaticMaskGenerator]: SamAutomaticMaskGenerator instance 
        """
        sam2 = build_sam2(self.model_config, self.model_weights, device=self.device, apply_postprocessing=False)
        mask_generator = SAM2AutomaticMaskGenerator(sam2)
        return mask_generator
            
    def bounding_boxes(self,mask: List[Dict],img: np.ndarray) -> np.ndarray:
        """
        Extracts bouding boxes from SAM masks

        Arguments:
            mask(List[Dict]): List of dictionaries including masks
            img(np.ndarray): An array of the original image

        Returns:
            np.ndarray: An array of the cropped image that fits the bounding box
        """

        self.x1 = int(mask['bbox'][0])
        self.y1 = int(mask['bbox'][1])
        self.length = self.x1 + int(mask['bbox'][2])
        self.height = self.y1 + int(mask['bbox'][3])        
        cropped_img = img[self.y1:self.height,self.x1:self.length]
        return cropped_img

    def _annotation_init(self,filename: str, data: np.ndarray):
        """
        Initialize annotation file with the filename and image size

        Arguments:
            filename(str): A string with the filename of the original image
            data(np.ndarray): An array of the original image
        """

        self.annotation = {}
        self.annotation["images"] = {
                                        "file_name": filename,
                                        "height": data.shape[0],
                                        "width": data.shape[1],
                                    }
        self.annotation_list = []

    def write_annotation(self,cropped_img_path: str):
        """
        Adds new annotation to annotation file

        Arguments:
            cropped_img_path(str): A string of the path for the cropped image file location
        """

        annoation = {
                        "filename": cropped_img_path,
                        "class_id": 0,
                        "segmentation" : self.img_metadata["segmentation"][self.y1:self.height,self.x1:self.length].tolist(),
                        "bbox" : self.img_metadata["bbox"]
                    }
        self.annotation_list.append(annoation)

    def save_annotation(self):
        """
        Writes annotation json file based on destination location on yaml file 
        """

        self.annotation["annotation"] = self.annotation_list
        json_filename = os.path.join(self.dest_directory,f'{os.path.basename(self.annotation["images"]["file_name"]).split(".")[0]}_.json')
        with open(json_filename, "w") as outfile:
            json_object = json.dumps(self.annotation)
            outfile.write(json_object)
    
    def get_masks(self,mask_generator: Type[SAM2AutomaticMaskGenerator] ,img_path: str) -> Tuple[np.ndarray,List[Dict]]:
        """
        Runs image through SAM mask generator 

        Arguments:
            mask_generator(Type[SamAutomaticMaskGenerator]): SAM mask_generator instance
            img_path(str): A string of the path of the image

        Returns:
            Tuple[np.ndarray,List[Dict]]: Returns an array of the original image and a list of dictionaries including masks
        """

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
        return (data, masks)

    def run(self):
        mask_generator = self.load_model()
        root_path = self.source_directory
        for img_path in tqdm(glob.glob(root_path + '/*.jpg')):
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
                self.write_annotation(cropped_img_path)
            self.save_annotation()
            print(f"file {img_path} is complete")
        self._update_cache()



