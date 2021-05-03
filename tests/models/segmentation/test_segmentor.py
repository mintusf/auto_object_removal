import os

# import pytest
import cv2
import numpy as np

from models.segmentation.segmentor import Segmentor


def test_segmentor():
    segmentor = Segmentor(os.path.join("config", "default.yaml"))

    input_image = cv2.imread(os.path.join("samples", "images", "dog_bicycle_car.jpg"))

    model_output = segmentor.predict_mask(input_image)

    mask_dog = segmentor.get_mask_sem(model_output, "dog")
    mask_bicycle = segmentor.get_mask_sem(model_output, "bicycle")
    mask_car = segmentor.get_mask_sem(model_output, "car")

    for mask in [mask_dog, mask_bicycle, mask_car]:
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (input_image.shape[0], input_image.shape[1])
        assert mask.sum() > 0