import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from typing import Tuple

from models.inpainting.abstract_inpainting_model import AbstractInpaintClass


class CRFillModel(AbstractInpaintClass):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self._build_model()

    def _build_model(self) -> None:
        self.model = InpaintGenerator()
        self.model.load_state_dict(torch.load(self.config["crfill"]["weights_path"]))
        self.model = self.model.to(self.device)

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

        # Change vcolor format
        img = img_orig[:, :, ::-1].copy()

        # COnvert to torch.tensor
        img = transforms.ToTensor()(img).unsqueeze(0)
        mask = transforms.ToTensor()(mask_orig).unsqueeze(0)

        # Normalize
        img = (img - 0.5) / 0.5

        # Send to devide
        img = img.to(self.device)
        mask = mask.to(self.device)

        return img, mask

    def _inference(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Performs inpainting inference

        Args:
            img (torch.Tensor): Preprocessed image
            mask (torch.Tensor): Preprocessed mask

        Returns:
            torch.Tensor: Model's output
        """
        _, model_output = self.model(img * (1 - mask), mask)
        model_output = model_output * mask + img * (1 - mask)
        model_output = model_output * 0.5 + 0.5

        return model_output

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

        h_raw, w_raw, _ = img_orig.shape
        img_orig = img_orig[:, :, ::-1]
        if len(mask_orig.shape) == 2:
            mask_orig = np.expand_dims(mask_orig, 2)

        if mask_orig.dtype == np.uint8:
            mask_orig = mask_orig / 255

        model_output = model_output.detach().cpu()[0].numpy() * 255
        model_output = model_output.transpose((1, 2, 0)).astype(np.uint8)
        model_output = cv2.resize(model_output, (w_raw, h_raw))
        inpaint_result = model_output * mask_orig + img_orig * (1 - mask_orig)
        inpaint_result = inpaint_result[:, :, ::-1]

        return inpaint_result


# The code from this repo is used here: https://github.com/zengxianyu/crfill
class InpaintGenerator(nn.Module):
    def __init__(self, cnum=48):
        super(InpaintGenerator, self).__init__()
        self.cnum = cnum
        # stage1
        self.conv1 = gen_conv(5, cnum, 5, 1)
        self.conv2_downsample = gen_conv(int(cnum / 2), 2 * cnum, 3, 2)
        self.conv3 = gen_conv(cnum, 2 * cnum, 3, 1)
        self.conv4_downsample = gen_conv(cnum, 4 * cnum, 3, 2)
        self.conv5 = gen_conv(2 * cnum, 4 * cnum, 3, 1)
        self.conv6 = gen_conv(2 * cnum, 4 * cnum, 3, 1)
        self.conv7_atrous = gen_conv(2 * cnum, 4 * cnum, 3, rate=2)
        self.conv8_atrous = gen_conv(2 * cnum, 4 * cnum, 3, rate=4)
        self.conv9_atrous = gen_conv(2 * cnum, 4 * cnum, 3, rate=8)
        self.conv10_atrous = gen_conv(2 * cnum, 4 * cnum, 3, rate=16)
        self.conv11 = gen_conv(2 * cnum, 4 * cnum, 3, 1)
        self.conv12 = gen_conv(2 * cnum, 4 * cnum, 3, 1)
        self.conv13_upsample_conv = gen_deconv(2 * cnum, 2 * cnum)
        self.conv14 = gen_conv(cnum, 2 * cnum, 3, 1)
        self.conv15_upsample_conv = gen_deconv(cnum, cnum)
        self.conv16 = gen_conv(cnum // 2, cnum // 2, 3, 1)
        self.conv17 = gen_conv(cnum // 4, 3, 3, 1, activation=None)

        # stage2
        self.xconv1 = gen_conv(3, cnum, 5, 1)
        self.xconv2_downsample = gen_conv(cnum // 2, cnum, 3, 2)
        self.xconv3 = gen_conv(cnum // 2, 2 * cnum, 3, 1)
        self.xconv4_downsample = gen_conv(cnum, 2 * cnum, 3, 2)
        self.xconv5 = gen_conv(cnum, 4 * cnum, 3, 1)
        self.xconv6 = gen_conv(2 * cnum, 4 * cnum, 3, 1)
        self.xconv7_atrous = gen_conv(2 * cnum, 4 * cnum, 3, rate=2)
        self.xconv8_atrous = gen_conv(2 * cnum, 4 * cnum, 3, rate=4)
        self.xconv9_atrous = gen_conv(2 * cnum, 4 * cnum, 3, rate=8)
        self.xconv10_atrous = gen_conv(2 * cnum, 4 * cnum, 3, rate=16)
        self.pmconv1 = gen_conv(3, cnum, 5, 1)
        self.pmconv2_downsample = gen_conv(cnum // 2, cnum, 3, 2)
        self.pmconv3 = gen_conv(cnum // 2, 2 * cnum, 3, 1)
        self.pmconv4_downsample = gen_conv(cnum, 4 * cnum, 3, 2)
        self.pmconv5 = gen_conv(2 * cnum, 4 * cnum, 3, 1)
        self.pmconv6 = gen_conv(2 * cnum, 4 * cnum, 3, 1, activation=nn.ReLU())
        self.pmconv9 = gen_conv(2 * cnum, 4 * cnum, 3, 1)
        self.pmconv10 = gen_conv(2 * cnum, 4 * cnum, 3, 1)

        self.allconv11 = gen_conv(4 * cnum, 4 * cnum, 3, 1)
        self.allconv12 = gen_conv(2 * cnum, 4 * cnum, 3, 1)
        self.allconv13_upsample_conv = gen_deconv(2 * cnum, 2 * cnum)
        self.allconv14 = gen_conv(cnum, 2 * cnum, 3, 1)
        self.allconv15_upsample_conv = gen_deconv(cnum, cnum)
        self.allconv16 = gen_conv(cnum // 2, cnum // 2, 3, 1)
        self.allconv17 = gen_conv(cnum // 4, 3, 3, 1, activation=None)

    def forward(self, x, mask):
        xin = x
        bsize, ch, height, width = x.shape
        ones_x = torch.ones(bsize, 1, height, width).to(x.device)
        x = torch.cat([x, ones_x, ones_x * mask], 1)

        # two stage network
        ## stage1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13_upsample_conv(x)
        x = self.conv14(x)
        x = self.conv15_upsample_conv(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = torch.tanh(x)
        x_stage1 = x

        x = x * mask + xin[:, 0:3, :, :] * (1.0 - mask)
        xnow = x

        ###
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x

        ###
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)

        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], 1)

        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample_conv(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample_conv(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.tanh(x)

        return x_stage1, x_stage2


class gen_conv(nn.Conv2d):
    def __init__(self, cin, cout, ksize, stride=1, rate=1, activation=nn.ELU()):
        """Define conv for generator

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
            Stride: Convolution stride.
            rate: Rate for or dilated conv.
            activation: Activation function after convolution.
        """
        p = int(rate * (ksize - 1) / 2)
        super(gen_conv, self).__init__(
            in_channels=cin,
            out_channels=cout,
            kernel_size=ksize,
            stride=stride,
            padding=p,
            dilation=rate,
            groups=1,
            bias=True,
        )
        self.activation = activation

    def forward(self, x):
        x = super(gen_conv, self).forward(x)
        if self.out_channels == 3 or self.activation is None:
            return x
        x, y = torch.split(x, int(self.out_channels / 2), dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y
        return x


class gen_deconv(gen_conv):
    def __init__(self, cin, cout):
        """Define deconv for generator.
        The deconv is defined to be a x2 resize_nearest_neighbor operation with
        additional gen_conv operation.

        Args:
            cin: Input Channel number.
            cout: output Channel number.
            ksize: Kernel size.
        """
        super(gen_deconv, self).__init__(cin, cout, ksize=3)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = super(gen_deconv, self).forward(x)
        return x