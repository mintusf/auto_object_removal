import os

# import pytest
import cv2

from models.segmentation.segmentor import Segmentor

def test_segmentor():
    segmentor = Segmentor(os.path.join("config", "default.yaml"))

    input_image = cv2.imread(os.path.join("samples", "images", "dog_bicycle_car.jpg"))

    mask_dog = segmentor.get_mask_sem(input_image, "dog")
    mask_bicycle = segmentor.get_mask_sem(input_image, "bicycle")
    mask_car = segmentor.get_mask_sem(input_image, "car")