import torchvision
from torchvision import transforms
import numpy as np

from models.segmentation import AbstractInstanceSegmentation

labels = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class InstSegMaskRcnn(AbstractInstanceSegmentation):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.confidence_threshold = self.config["instance_confidence_threshold"]
        self._build_model()

        self.classid2label = dict([(i, label) for (i, label) in enumerate(labels)])

    def _build_model(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    def _preprocess(self, img_orig):
        transform = transforms.Compose([transforms.ToTensor()])

        input_image = transform(img_orig)

        return input_image.unsqueeze(0)

    def _inference(self, img):
        model_output = self.model(img)
        return model_output

    def _postprocess(self, model_output, img_orig):

        labels = model_output[0]["labels"].detach().numpy()
        scores = model_output[0]["scores"].detach().numpy()
        masks = (model_output[0]["masks"] > 0.5).detach().numpy()
        masks = np.transpose(masks, [0, 2, 3, 1])

        high_confidence_instance = scores > self.confidence_threshold
        masks = masks[high_confidence_instance]
        labels = labels[high_confidence_instance]

        return masks, labels
