import yaml
import wget
import tarfile
import pdb
from utils import isEmpty,setupdata,isPresent
from cocosplit import splitdata
from ObjectDetector import DetectronDetector
def main(*args, **kwargs):
    with open("config.yaml") as f:
        config = yaml.load(f, Loader = yaml.SafeLoader)

    dataset = config['Dataset']
    print('setting up')
    setupdata(dataset_config = dataset)
    print('setup done')
    ## split train and val
    #FIXME
    # will move it to config files
    splitdata(filepath = r"data/trainval/annotations/bbox-annotations.json",is_annotated = True,split_ratio = 0.8)
    detector = DetectronDetector().setup(datadir = '/data/trainval',trainannotations = 'data/trainval/annotations/train.json',testannotations =  'data/trainval/annotations/test.json', imgpath = 'data/trainval/images' )
    if not isPresent('output/model_final.pth'): # just to save time of checking and creating models dir
      detector.train()
    detector.run_inference()
if __name__ == "__main__":
    main()