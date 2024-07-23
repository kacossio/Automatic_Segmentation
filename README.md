# Automatic_Segmentation

This project will segment images using facebook's Segment Anything model and write the segmented images and annotaions to a folder. 

You can view facebooks github here: https://github.com/facebookresearch/segment-anything

To run this model you must install all relevant packages. 

There must also be a yaml folder with file locations, model weights, and cache file location. Below is an example yaml file

```yaml
---
source_directory: data                          #The directory with all jpg images that are too be segmented
dest_directory: data/cropped                    #destanation directory of segmented images
model_weights: weights/sam_vit_h_4b8939.pth     #location of model weights
cache_file: _cached_mask.pickle                 #optional cached file
```