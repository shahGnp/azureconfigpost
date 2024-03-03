import torch

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from matplotlib import pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode



# if "my_dataset_train" in detectron2.data.DatasetCatalog.list():
#     detectron2.data.DatasetCatalog.remove("my_dataset_train")
#     detectron2.data.MetadataCatalog.remove("my_dataset_train")

if "my_dataset_val" in detectron2.data.DatasetCatalog.list():
    detectron2.data.DatasetCatalog.remove("my_dataset_val")
    detectron2.data.MetadataCatalog.remove("my_dataset_val")

# register_coco_instances("my_dataset_train", {}, "./dataset/combined_xview_instance_segmentation_dataset_train.json", "./dataset/")
register_coco_instances("my_dataset_val", {}, "./dataset/combined_xview_damage_assessment_instance_segmentation_dataset_val.json", "./dataset/")

# train_metadata = MetadataCatalog.get("my_dataset_train")
# train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

val_metadata = MetadataCatalog.get("my_dataset_val")
# val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("my_dataset_train",)
# cfg.DATASETS.TEST = ("my_dataset_val")
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.MODEL.DEVICE='cpu'
cfg.SOLVER.MAX_ITER = 50000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

cfg.MODEL.WEIGHTS = os.path.join('./models/', "model_final_post.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set a custom testing threshold
predictor = DefaultPredictor(cfg)


def detectImage(image_path):
    im = cv2.imread(os.path.join('./',image_path))
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=val_metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])
    cv2.imwrite('./detection_post.png',out.get_image()[:, :, ::-1])
    




# detectImage('./guatemala-volcano_00000000_pre_disaster.png')
