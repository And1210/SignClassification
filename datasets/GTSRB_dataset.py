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
    "speed_20": 0,
    "speed_30": 1,
    "speed_50": 2,
    "speed_60": 3,
    "speed_70": 4,
    "speed_80": 5,
    "speed_80_alt": 6,
    "speed_100": 7,
    "speed_120_alt": 8,
    "yield": 13,
    "stop": 14
}

SIGN_DICT_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    13: 9,
    14: 10
}

class GTSRBDataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]

        self._image_size = tuple(configuration["input_size"])

        self.dataset_path = configuration["dataset_path"]
        self.annotations = pd.read_csv(os.path.join(configuration["dataset_path"], "{}.csv".format(self._stage)))

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):
        # i = index
        # self._data = ET.parse(os.path.join(self.dataset_path, "annotations/road{}.xml".format(i)))
        # filename = self._data.findall('./filename')[0].text
        ann = np.array(list(self.annotations.iloc[index]))

        pixels = Image.open(os.path.join(self.dataset_path, "{}".format(ann[7]))).convert('RGBA')
        image = np.asarray(pixels)#.reshape(48, 48)
        image = image.astype(np.uint8)

        cropped = image[int(ann[3]):int(ann[5]), int(ann[2]):int(ann[4])]

        # print(self._image_size)

        # print(x, y, w, h)
        # print(cropped.shape)
        # plt.imshow(cropped)
        # plt.show()
        try:
            image = cv2.resize(cropped, self._image_size)
        except:
            # print(x, y, w, h)
            print(ann[2:6])
            print(cropped.shape)
            plt.imshow(cropped)
            plt.show()

        image = np.dstack([image] * 1)
        # image = np.dstack([image] * 3)

        # if self._stage == "train":
        # image = seg(image=image)

        label = int(ann[6])

        image = self._transform(image)
        if (label in list(SIGN_DICT_MAP.keys())):
            target = SIGN_DICT_MAP[label]
        else:
            target = 11
        return image, target

    def __len__(self):
        # return the size of the dataset
        return len(self.annotations)

    def get_label(self, label):
        index = list(SIGN_DICT_MAP.keys())[label]
        for d in SIGN_DICT:
            if SIGN_DICT[d] == index:
                return d
        return "Other"
