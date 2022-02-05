import detectron2
from pathlib import Path
import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.evaluation.evaluator import DatasetEvaluator

from detectron2.modeling import build_model

from detectron2.checkpoint import DetectionCheckpointer
import torch

## Class to setup detectron
class DetectronDetector:
    dataDir=Path('/data/trainval')
    cfg = get_cfg()
    register_coco_instances('task_train',{}, 'data/trainval/annotations/train.json', 'data/trainval/images')
    register_coco_instances('task_val',{},'data/trainval/annotations/test.json', 'data/trainval/images')
    metadata = MetadataCatalog.get('task_train')
    train_ds = DatasetCatalog.get('task_train')
    

    def train(self):
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
        self.cfg.DATASETS.TRAIN = ("task_train",)
        self.cfg.DATASETS.TEST = ()
        # self.cfg.DATASETS.TEST = ("task_val",)
        # Number of data loading threads
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        # Number of images per batch across all machines.
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.000525  # pick a good LearningRate
        self.cfg.SOLVER.MAX_ITER = 5000  #No. of iterations   
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # No. of classes = [PERSON, CAR]
        self.cfg.TEST.EVAL_PERIOD = 500 # No. of iterations after which the Validation Set is evaluated. 
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg) 
        trainer.resume_or_load(resume=False)
        trainer.train()
        model = build_model(self.cfg)  # returns a torch.nn.Module
        if wanna_save:
            checkpointer = DetectionCheckpointer(model, save_dir="output")
            checkpointer.save("model_5000")  # save to output/model_999.pth
            torch.save(model.state_dict(), 'checkpoint.pth')

    def predict_small_train_sample(self):
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
        predictor = DefaultPredictor(self.cfg)
        dataset_dicts = DatasetCatalog.get('task_val')
        outs = []
        for d in random.sample(dataset_dicts, 3):    
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata = MetadataCatalog.get('task_train'), 
                            
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            visualizer = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get('task_train'))
            out_target = visualizer.draw_dataset_dict(d)
            outs.append(out_pred)
            outs.append(out_target)
        _,axs = plt.subplots(len(outs)//2,2,figsize=(40,45))
        for ax, out in zip(axs.reshape(-1), outs):
            ax.imshow(out.get_image())

    def evaluate():
        #import the COCO Evaluator to use the COCO Metrics
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        #Call the COCO Evaluator function and pass the Validation Dataset
        evaluator = COCOEvaluator("task_val", self.cfg, False, output_dir="/output/")
        val_loader = build_detection_test_loader(self.cfg, "task_val")

        #Use the created predicted model in the previous step
        inference_on_dataset(predictor.model, val_loader, evaluator)