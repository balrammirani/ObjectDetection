import yaml
import wget
import tarfile
import pdb
from utils import isEmpty,setupdata
from cocosplit import splitdata
from ObjectDetector import DetectronDetector
def main(*args, **kwargs):
    with open("config.yaml") as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)
    dataset = config['Dataset']
    setupdata(dataset_config = dataset)
    ## split train and val
    splitdata(filepath = r"data/trainval/annotations/bbox-annotations.json",is_annotated = True,split_ratio = 0.8)
    detector = DetectronDetector()
    if isEmpty('models'):
        detector.train()
    else:
        detector.evaluate()
if __name__ == "__main__":
    main()