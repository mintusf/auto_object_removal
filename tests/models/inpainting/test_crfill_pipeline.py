import os

# import pytest
import cv2
import numpy as np

from models.segmentation.segmentor import Segmentor
from models.inpainting import CRFillModel


def test_crfill_pipeline():
    config_path = os.path.join("config", "default.yaml")
    segmentor = Segmentor(config_path)
    inpainter = CRFillModel(config_path, pretrained=False)

    input_image = cv2.imread(
        os.path.join("samples", "images", "dog_bicycle_car.jpg"), cv2.IMREAD_COLOR
    )

    model_output = segmentor.predict_mask(input_image)

    mask_dog = segmentor.get_mask_sem(model_output, ["dog"])
    mask_bicycle = segmentor.get_mask_sem(model_output, ["bicycle"])
    mask_car = segmentor.get_mask_sem(model_output, ["car"])

    for i, mask in enumerate([mask_dog, mask_bicycle, mask_car]):
        inpainted_result = inpainter.inpaint(input_image, mask)

        assert isinstance(inpainted_result, np.ndarray)
        assert len(inpainted_result.shape) == 3
        assert inpainted_result.shape == input_image.shape
        assert (inpainted_result != input_image).any()
