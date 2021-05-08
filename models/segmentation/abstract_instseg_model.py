import abc
import torch
import numpy as np
from typing import Tuple

from utils.load import parse_config

class AbstractInstanceSegmentation(metaclass=abc.ABCMeta):

    def __init__(self, config_path):
        self.config = parse_config(config_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def _buil_model(self) -> None:
        raise NotImplementedError


    @abc.abstractmethod
    def _preprocess(
        self, img_orig: np.array
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def _postprocess(
        self, model_output: torch.Tensor, img_orig: np.array
    ) -> np.array:
        """Performs postprocessing to obtain inpainted results

        Args:
            model_output (torch.Tensor): Model's output
            img_orig (np.array): Input image

        Returns:
            np.array: Inpainting results
        """
        raise NotImplementedError

    def predict(self, img_orig: np.array) -> Tuple(np.array, list):
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

        masks, labels = self._postprocess(model_output, img_orig,)

        return masks, labels