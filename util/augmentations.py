import numpy as np

def resize_img_bbox(img_shape, bbox,target_size):
    x_ratio = target_size[0] / img_shape[0]
    y_ratio = target_size[1] / img_shape[1]

    y = int(np.round(bbox[0]*y_ratio))
    x = int(np.round(bbox[1]*x_ratio))
    y_max = int(np.round(bbox[2]*y_ratio))
    x_max = int(np.round(bbox[3]*x_ratio))

    return [y,x,y_max,x_max]



