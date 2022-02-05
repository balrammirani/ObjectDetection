# ObjectDetection

Trained on OpenImages data. Aim is to detect humans and cars in the test set. I have used detectron2 framework and its faster-rcnn model_config for detecting bounding boxes.

Following is the summary of this experiment:
1. Used a modified version of cocosplit package to split the dataset into train and validation
2. Used "faster_rcnn_R_50_FPN_3x" as a starter to validate the process.
3. Used a universal_seed_set module to maintain the reproducibility. The module works for every aspect except the training nn_module. Someone suggested to use "torch.backends.cudnn.benchmark = True" but th results were still random.
4. Used simple parameters to train the model. Tuning may help in improving the accuracy.
5. Data Augmentation is not touched however it will be added in future iterations.
6. Evaluation results:

|   AP   |  AP50  |  AP75  |  APs   |  APm  |  APl  |
|:------:|:------:|:------:|:------:|:-----:|:-----:|
| 40.110 | 69.250 | 41.345 | 48.001 |  nan  |  nan  |

| category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|
| person     | 34.029 | car        | 46.191 |


