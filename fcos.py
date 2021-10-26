import random

import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
import os

from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

register_coco_instances("train",
                        {},
                        "D:/projects_python/_datasets/coco_2017/annotations/instances_train2017.json",
                        "D:/projects_python/_datasets/coco_2017/train2017/")

register_coco_instances("val",
                        {},
                        "D:/projects_python/_datasets/coco_2017/annotations/instances_val2017.json",
                        "D:/projects_python/_datasets/coco_2017/val2017/")
