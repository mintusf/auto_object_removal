import abc
from typing import Tuple
import cv2
import numpy as np
import torch

from utils.load import parse_config


class AbstractInpaintClass(metaclass=abc.ABCMeta):
    def __init__(self, config_path):
        self.config = parse_config(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def _build_model(self) -> None:
        """Builds a model for inpainting"""
        raise NotImplementedError

    @abc.abstractmethod
    def _preprocess(
        self, img_orig: np.array, mask_orig: np.array
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess input image and mask to prepare for inference

        Args:
            img_orig (np.array): Input image
            mask_orig (np.array): Input mask indicating area to inpaint

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Preprocesses image and mask
        """

        raise NotImplementedError

    @abc.abstractmethod
    def _inference(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Performs inpainting inference

        Args:
            img (torch.Tensor): Preprocessed image
            mask (torch.Tensor): Preprocessed mask

        Returns:
            torch.Tensor: Model's output
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _postprocess(
        self, model_output: torch.Tensor, img_orig: np.array, mask_orig: np.array
    ) -> np.array:
        """Performs postprocessing to obtain inpainted results

        Args:
            model_output (torch.Tensor): Model's output
            img_orig (np.array): Input image
            mask_orig (np.array): Input mask indicating area to inpaint

        Returns:
            np.array: Inpainting results
        """
        raise NotImplementedError

    def inpaint(self, img_orig: np.array, mask_orig: np.array) -> np.array:
        """Given input image and mask, performs inpainting

        Args:
            img_orig (np.array): Input image
            mask_orig (np.array): Input mask indicating area to inpaint

        Returns:
            np.array: [description]
        """

        img, mask = img_orig.copy(), mask_orig.copy()

        img, mask = self._preprocess(img, mask)

        model_output = self._inference(img, mask)

        inpainted_result = self._postprocess(model_output, img_orig, mask_orig)

        return inpainted_result