import os

# import pytest
import cv2
import numpy as np

from models.segmentation.instseg_maskrcnn import InstSegMaskRcnn


def test_segmentor():
    segmentor = InstSegMaskRcnn(os.path.join("config", "default.yaml"))

    input_image = cv2.imread(os.path.join("samples", "images", "human_multi.jpg"))

    masks, labels = segmentor.predict(input_image)

    available_labels = segmentor.detect_labels(masks, labels, 419, 468)

    assert segmentor.get_mask(masks, labels, ["troll"], 419, 468) is None

    combined_mask = segmentor.get_mask(masks, labels, ["backpack"], 419, 468)
    cv2.imwrite(f"combined_mask.png", np.where(combined_mask, 0, input_image))

    for i in range(masks.shape[0]):
        mask = masks[i, :, :].astype(np.uint8)
        label = labels[i]
        masked = np.where(mask, 0, input_image)
        cv2.imwrite(f"mask_{i}_{segmentor.classid2label[label]}.png", masked)
