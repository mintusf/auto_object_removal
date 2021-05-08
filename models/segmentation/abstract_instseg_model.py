import abc
import torch
import numpy as np
from typing import Tuple
import cv2

from utils.load import parse_config


class AbstractInstanceSegmentation(metaclass=abc.ABCMeta):
    def __init__(self, config_path):
        self.config = parse_config(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mask_dilation_size = self.config["mask_dilation_size"]
        self.mask_opening_size = self.config["mask_opening_size"]

    @abc.abstractmethod
    def _build_model(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _preprocess(self, img_orig: np.array) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess input image to prepare for inference

        Args:
            img_orig (np.array): Input image

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Preprocesses image
        """

        raise NotImplementedError

    @abc.abstractmethod
    def _inference(self, img: torch.Tensor) -> torch.Tensor:
        """Performs inpainting inference

        Args:
            img (torch.Tensor): Preprocessed image

        Returns:
            torch.Tensor: Model's output
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _postprocess(self, model_output: torch.Tensor, img_orig: np.array) -> np.array:
        """Performs postprocessing to obtain inpainted results

        Args:
            model_output (torch.Tensor): Model's output
            img_orig (np.array): Input image

        Returns:
            np.array: Inpainting results
        """
        raise NotImplementedError

    def predict(self, img_orig: np.array) -> Tuple[np.array, list]:
        """Given input image and mask, performs inpainting

        Args:
            img_orig (np.array): Input image

        Returns:
            Tuple containing:
            * (np.array): containing all detected masks
            * (list): containing labels corresponding to masks
        """

        img = img_orig.copy()

        img = self._preprocess(img)

        model_output = self._inference(img)

        masks, labels = self._postprocess(model_output, img_orig)

        return masks, labels

    def _get_mask_label_at_coordinates(self, masks, labels, x, y):
        # Get all available masks at this coordinates
        available_instances_idx = [masks[i, y, x, 0] > 0 for i in range(masks.shape[0])]
        available_masks = masks[available_instances_idx]
        available_labels = [
            self.classid2label[label] for label in labels[available_instances_idx]
        ]

        return available_masks, available_labels

    def detect_labels(self, masks, labels, x, y):

        _, available_labels = self._get_mask_label_at_coordinates(masks, labels, x, y)

        return np.unique(available_labels)

    def get_mask(self, masks, labels, selected_labels, x, y):

        available_masks, available_labels = self._get_mask_label_at_coordinates(
            masks, labels, x, y
        )

        used_idx = []
        for label in selected_labels:
            label_first_idx = np.where(np.array(available_labels) == np.array(label))[
                0
            ][0]
            used_idx.append(label_first_idx)

        used_masks = available_masks[used_idx]
        mask = np.apply_along_axis(lambda arr: arr.any(), 0, used_masks).astype(
            np.uint8
        )

        # Morphology
        # Opening
        opening_kernel = np.ones(
            (self.mask_opening_size, self.mask_opening_size), np.uint8
        )
        mask = cv2.dilate(mask, opening_kernel)
        mask = cv2.erode(mask, opening_kernel)

        # Dilation
        dilation_kernel = np.ones(
            (self.mask_dilation_size, self.mask_dilation_size), np.uint8
        )
        mask = cv2.dilate(mask, dilation_kernel)

        return np.expand_dims(mask, 2)
