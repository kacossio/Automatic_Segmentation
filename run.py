import util.extract_segmentation

if __name__=="__main__":

    Segmenter = util.extract_segmentation.Segmenter(config="config.yaml")
    Segmenter.run()