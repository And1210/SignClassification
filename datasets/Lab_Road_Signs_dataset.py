import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
from utils.augmenters.augment import seg
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt

SIGN_DICT = {
    "other": 0,
    "speedlimit": 1,
    "stop": 2
}

class LabRoadSignsDataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]

        self._image_size = tuple(configuration["input_size"])

        self.dataset_path = os.path.join(configuration["dataset_path"], "{}".format(self._stage))

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):
        i = index
        self._data = ET.parse(os.path.join(self.dataset_path, "annotations/sign_{}.xml".format(i)))
        filename = self._data.findall('./filename')[0].text

        pixels = Image.open(os.path.join(self.dataset_path, "images/{}".format(filename))).convert('RGBA')
        image = np.asarray(pixels)#.reshape(48, 48)
        image = image.astype(np.uint8)

        image = cv2.resize(image, self._image_size)

        image = np.dstack([image] * 1)
        # image = np.dstack([image] * 3)

        # if self._stage == "train":
        # image = seg(image=image)

        label = self._data.findall('./name')[0].text

        image = self._transform(image)
        target = SIGN_DICT[label]
        return image, target

    def __len__(self):
        # return the size of the dataset
        return len(os.listdir(os.path.join(self.dataset_path, 'annotations')))

    def get_label(self, label):
        return list(SIGN_DICT.keys())[label]
