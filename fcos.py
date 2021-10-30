import random

import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
import os

from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

import torch.nn as nn


@BACKBONE_REGISTRY.register()
class ToyBackBone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        # create your own backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, image):
        return {"conv1": self.conv1(image)}

    def output_shape(self):
        return {"conv1": ShapeSpec(channels=64, stride=1)}


if __name__ == '__main__':
    register_coco_instances("my_dataset_train",
                            {},
                            "D:/projects_python/_datasets/coco_2017/annotations/instances_train2017.json",
                            "D:/projects_python/_datasets/coco_2017/train2017/")

    register_coco_instances("my_dataset_val",
                            {},
                            "D:/projects_python/_datasets/coco_2017/annotations/instances_val2017.json",
                            "D:/projects_python/_datasets/coco_2017/val2017/")

    metadata = MetadataCatalog.get("my_dataset_train")
    dataset_dicts = DatasetCatalog.get("my_dataset_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00005
    # cfg.SOLVER.MAX_ITER = 300
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = ("my_dataset_val",)
    predictor = DefaultPredictor(cfg)

    im = cv2.imread("D:/projects_python/_datasets/coco_2017/train2017/000000000036.jpg")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

'''
im = cv2.imread("D:/projects_python/_datasets/coco_2017/train2017/000000000036.jpg")

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow('', out.get_image()[:, :, ::-1])
cv2.waitKey(0)
'''
