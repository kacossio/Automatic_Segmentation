import util.segmentation

if __name__ == "__main__":
    segmenter = util.segmentation.Segmenter(config="config.yaml")
    segmenter.run()