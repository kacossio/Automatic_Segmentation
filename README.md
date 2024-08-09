# Automatic_Segmentation

This project will segment images using facebook's Segment Anything model and write the segmented images and annotaions to a folder. 

## Installation

I am running the nvidia pytorch container that matches my cuda setup. You can find the correct container here: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

Once you have the correct container/enviroment, install segment anyting using steps from facebooks here: https://github.com/facebookresearch/segment-anything
Make sure to also download the weights to the model you will be using.

## Usage

Create a yaml file with file locations, model weights, and cache file location. Below is an example yaml file

```yaml
---
source_directory: data                          #The directory with all jpg images that are too be segmented
dest_directory: data/cropped                    #destanation directory of segmented images
model_weights: weights/sam_vit_h_4b8939.pth     #location of model weights
cache_file: _cached_mask.pickle                 #optional cached file
```

Afterwards run the following command:

```python
python3 run.py
```