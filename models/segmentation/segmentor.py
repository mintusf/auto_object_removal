from typing import Tuple
import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2

from utils.load import parse_config


class Segmentor:
    def __init__(self, config_path: str):

        # Parse config parameters
        self.config = parse_config(config_path)
        self.max_instances = self.config["max_instances"]
        self.semantic_segmentation_model_cfg = self.config["semantic_segmentation_cfg"]

        self.background_class_threshold = self.config["background_class_threshold"]
        self.foreground_class_threshold = self.config["foreground_class_threshold"]
        self.mask_dilation_size = self.config["mask_dilation_size"]
        self.mask_opening_size = self.config["mask_opening_size"]

        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._build_models()

    def _build_models(self) -> None:
        """Build semantic and instance (TODO) segmentation models as well preprocessing and postprocessing parameters"""

        ### SEMANTIC SEGMENTATION ###
        if self.semantic_segmentation_model_cfg[1] == "deeplab":
            if self.semantic_segmentation_model_cfg[0] == "resnet50":
                self.semseg_model = torchvision.models.segmentation.deeplabv3_resnet50(
                    pretrained=True
                )
            elif self.semantic_segmentation_model_cfg[0] == "resnet101":
                self.semseg_model = torchvision.models.segmentation.deeplabv3_resnet101(
                    pretrained=True
                )
            elif self.semantic_segmentation_model_cfg[0] == "mobilenet":
                self.semseg_model = (
                    torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
                        pretrained=True
                    )
                )
            else:
                raise NotImplementedError(
                    "Unsupported semantic segmentation model config"
                )
        else:
            raise NotImplementedError("Only deeplab model is supported for now")

        # Move model to a correct device and set to eval mode
        self.semseg_model = self.semseg_model.to(self.device)
        self.semseg_model.eval()

        # Create classname-channel pairs of model's output
        self._parse_class_idx()

        # Normalization parameters
        if self.semantic_segmentation_model_cfg[1] == "deeplab":
            self._semseg_mean = [0.485, 0.456, 0.406]
            self._semseg_std = [0.229, 0.224, 0.225]
        else:
            raise NotImplementedError

    @property
    def semseg_mean(self):
        return self._semseg_mean

    @property
    def semseg_std(self):
        return self._semseg_std

    def _parse_class_idx(self) -> None:
        """Parses indices (channels) of corresponding classes in final prediction"""

        # Parse for semantic segmentation
        self.semseg_class2channel_list = self.config["semseg_idx"][
            self.semantic_segmentation_model_cfg[1]
        ]
        self.semseg_class2channel_dict = dict(
            [
                (k, v)
                for (k, v) in zip(
                    self.semseg_class2channel_list,
                    np.arange(0, len(self.semseg_class2channel_list)),
                )
            ]
        )

        self.semseg_channel2class_dict = dict(
            [
                (v, k)
                for (k, v) in zip(
                    self.semseg_class2channel_list,
                    np.arange(0, len(self.semseg_class2channel_list)),
                )
            ]
        )

    def get_available_masks(self, mask: np.array) -> list:
        assert mask.max() <= len(
            self.semseg_class2channel_list
        ), "Shape of mask does not match selected model"

        available_channels = np.unique(mask)

        available_classes = [
            self.semseg_class2channel_list[channel]
            for channel in available_channels
            if channel != 0
        ]

        return available_classes

    def _preprocess(self, input_image: np.array) -> torch.tensor:
        """Performs preprocessing of input image

        Args:
            input_image (np.array): Image to be used as model's input

        Returns:
            torch.tensor: Preprocessed image ready for inference
        """

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.semseg_mean, self.semseg_std),
            ]
        )

        input_image = transform(input_image)

        return input_image.unsqueeze(0)

    def _mask2class(self, mask: torch.Tensor) -> np.array:
        """ Postprocess semantic model's output to assign class to each pixel

        Args:
            mask (torch.Tensor): Semantic segmentation model's output

        Returns:
            np.array: Array with a semantic class assigned to each pixel
        """

        mask = mask.cpu().numpy()

        # Get background channel
        background_channel = self.semseg_class2channel_dict["background"]

        # Apply softmax
        mask_exp = np.exp(mask)
        mask_normalized = mask_exp / mask_exp.sum(0)

        # Calculate where background is higher than threshold (softmax is used)
        low_background_confidence = (
            mask_normalized[background_channel, :, :] < self.background_class_threshold
        )

        # Calculate where any foreground class is higher than threshols
        foreground_classes_only = np.delete(mask_normalized, background_channel, 0)
        high_foreground_confidence = np.any(
            foreground_classes_only > self.foreground_class_threshold, axis=0
        )

        # Find class with highest probability (excluding background)
        mask_min = mask.min()
        mask[background_channel, :, :] = mask_min
        output_predictions = mask.argmax(0)

        # Set final class
        output_predictions = np.where(
            low_background_confidence & high_foreground_confidence,
            output_predictions,
            background_channel,
        )

        return output_predictions

    def predict_mask(self, input_image: np.array) -> np.array:
        """Performs inference on the image

        Args:
            input_image (np.array): Input image

        Returns:
            np.array: Predictions of size (C, H, W)
        """

        input_image = self._preprocess(input_image)

        # Generate a tensor which indicates pixel-wise class
        with torch.no_grad():
            output = self.semseg_model(input_image)["out"][0]

        output_predictions = self._mask2class(output)

        return output_predictions

    def get_mask_sem(self, output_predictions: np.array, class_name: str) -> np.array:
        """Using model's predictions, extract mask for a desired class

        Args:
            output_predictions (np.array): Model's predictions
            class_name (str): Name of the class which mask is to be extracted

        Returns:
            np.array: Mask of desired class
        """

        # class name check
        if class_name not in self.semseg_class2channel_list:
            raise ValueError(f"Class name {class_name} is not supported")

        # Get mask of desired class
        class_idx = self.semseg_class2channel_dict[class_name]
        class_mask = np.where(output_predictions == class_idx, 255, 0).astype(np.uint8)

        # Opening
        opening_kernel = np.ones(
            (self.mask_opening_size, self.mask_opening_size), np.uint8
        )
        class_mask = cv2.dilate(class_mask, opening_kernel)
        class_mask = cv2.erode(class_mask, opening_kernel)

        # Dilation
        dilation_kernel = np.ones(
            (self.mask_dilation_size, self.mask_dilation_size), np.uint8
        )
        class_mask = cv2.dilate(class_mask, dilation_kernel)

        return class_mask
