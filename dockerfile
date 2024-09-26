FROM nvcr.io/nvidia/pytorch:22.04-py3
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
