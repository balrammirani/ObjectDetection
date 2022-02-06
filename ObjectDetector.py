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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from tqdm  import tqdm
from utils import seed_everything


# Evaluation Metrics HElper.
# Will move it to helper class in future release
#FIXME
# Taken from https://www.kaggle.com/theoviel/competition-metric-map-iou
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    return np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

def score(pred, targ):
    pred_masks = pred['instances'].pred_masks.cpu().numpy()
    enc_preds = [mask_util.encode(np.asarray(p, order='F')) for p in pred_masks]
    enc_targs = list(map(lambda x:x['segmentation'], targ))
    ious = mask_util.iou(enc_preds, enc_targs, [0]*len(enc_targs))
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, ious)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)

class MAPIOUEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        dataset_dicts = DatasetCatalog.get(dataset_name)
        self.annotations_cache = {item['image_id']:item['annotations'] for item in dataset_dicts}
            
    def reset(self):
        self.scores = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if len(out['instances']) == 0:
                self.scores.append(0)    
            else:
                targ = self.annotations_cache[inp['image_id']]
                self.scores.append(score(out, targ))

    def evaluate(self):
        return {"MaP IoU": np.mean(self.scores)}

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return MAPIOUEvaluator(dataset_name)


## Class to setup detectron
class DetectronDetector:
    def __init__(self,*args,**kwargs):
        self.dataDir = None
        self.metadata = None
        self.train_ds = None
    @classmethod
    def setup(cls,datadir,trainannotations,testannotations,imgpath):
        register_coco_instances('task_train',{}, trainannotations, imgpath)
        register_coco_instances('task_val',{},testannotations, imgpath)
        
        detectronobject =  cls(dataDir = Path(datadir), metadata =MetadataCatalog.get('task_train'),train_ds = DatasetCatalog.get('task_train'))
        
        return detectronobject
    def train(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("task_train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.000525
        cfg.SOLVER.MAX_ITER = 5000    
        cfg.SOLVER.STEPS = []        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
        cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get('task_train')) // cfg.SOLVER.IMS_PER_BATCH  # Once per epoch
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        # model = build_model(self.cfg)  # returns a torch.nn.Module
        # if wanna_save:
        #     checkpointer = DetectionCheckpointer(model, save_dir="output")
        #     checkpointer.save("model_5000")  # save to output/model_5000.pth
        #     torch.save(model.state_dict(), os.path.join('output','checkpoint.pth')) ## just to see if torch.save and checkpointer are generating same value
        self.evaluate(cfg)

    def evaluate(self,cfg):        
        ## predictions
        print(cfg)
        cfg.MODEL.WEIGHTS = os.path.join('output',"model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        predictor = DefaultPredictor(cfg)
        #Call the COCO Evaluator function and pass the Validation Dataset
        evaluator = COCOEvaluator("task_val", cfg, False, output_dir="/output/")
        val_loader = build_detection_test_loader(cfg, "task_val")

        #Use the created predicted model in the previous step
        inference_on_dataset(predictor.model, val_loader, evaluator)
    
    def run_inference(self):
        ## predictions
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
        cfg.DATASETS.TRAIN = ("task_train",)
        cfg.DATASETS.TEST = ()
        # Number of data loading threads
        cfg.DATALOADER.NUM_WORKERS = 2
        
        cfg.MODEL.WEIGHTS = os.path.join('output',"model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        predictor = DefaultPredictor(cfg)
        dataset_dicts = DatasetCatalog.get('task_val')
        for d in tqdm(dataset_dicts):
            outs=[]
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],metadata = MetadataCatalog.get('task_val'))
            out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            visualizer = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get('task_val'))
            out_target = visualizer.draw_dataset_dict(d)
            outs.append(out_pred)
            outs.append(out_target)
            _,axs = plt.subplots(len(outs)//2,2,figsize=(40,45))
            for ax, out in zip(axs.reshape(-1), outs):
                ax.imshow(out.get_image())
            plt.savefig(os.path.join(cfg.OUTPUT_DIR,"pred_{}.jpg".format(d['image_id']))) # To save figure
            plt.close('all')
            plt.clf() 
            del outs,axs,v,visualizer,out_target,out_pred,im

    # Will modify it in future for side to side comparison of train dataset vs predicted on train data for random n inferences
    # def predict_small_train_sample(self):
    #     self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    #     self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    #     predictor = DefaultPredictor(self.cfg)
    #     dataset_dicts = DatasetCatalog.get('task_val')
    #     outs = []
    #     for d in random.sample(dataset_dicts, 3):    
    #         im = cv2.imread(d["file_name"])
    #         outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    #         v = Visualizer(im[:, :, ::-1],
    #                     metadata = MetadataCatalog.get('task_train'), 
                            
    #                     instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #         )
    #         out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #         visualizer = Visualizer(im[:, :, ::-1], metadata=MetadataCatalog.get('task_train'))
    #         out_target = visualizer.draw_dataset_dict(d)
    #         outs.append(out_pred)
    #         outs.append(out_target)
    #     _,axs = plt.subplots(len(outs)//2,2,figsize=(40,45))
    #     for ax, out in zip(axs.reshape(-1), outs):
    #         ax.imshow(out.get_image())