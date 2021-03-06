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
| 40.066 | 69.448 | 40.803 | 48.665 |  nan  |  nan  |

| category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|
| person     | 33.537 | car        | 46.596 |

Google Colab Link : https://colab.research.google.com/drive/17ZGzjC9M3RGWqBINtARSn-0-rKTWZQin

Sample Inference: (LEFT is Predicted and RIGHT is Actual) 

![1](https://user-images.githubusercontent.com/11017748/152732419-94278158-a0af-4620-a7fb-b068821dadac.png)
