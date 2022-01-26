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
import json
import torch

class LISADataset(BaseDataset):
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
        image_data = []
        for i in range(3):
            image_data.append(torch.load(os.path.join(self.dataset_path, 'images_{}.tensor'.format(i))))
        image_data = torch.cat(image_data, dim=0)
        self.images = []
        for i in range(len(image_data)):
            image = image_data[i].cpu().detach().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = np.insert(image, 3, 255, axis=2)
            self.images.append(image)
        self.labels = torch.load(os.path.join(self.dataset_path, 'labels.tensor'))
        with open(os.path.join(self.dataset_path, 'meta.js')) as f:
            self.classes = json.load(f)['classes']

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):
        # pixels = Image.open(os.path.join(self.dataset_path, "{}".format(ann[7]))).convert('RGBA')
        pixels = self.images[index]
        image = np.asarray(pixels)#.reshape(48, 48)
        image = image.astype(np.uint8)
        image = cv2.resize(image, self._image_size)

        image = np.dstack([image] * 1)
        # image = np.dstack([image] * 3)

        # if self._stage == "train":
        # image = seg(image=image)

        image = self._transform(image)
        target = self.labels[index]
        return image, target

    def __len__(self):
        # return the size of the dataset
        return len(self.labels)

    def get_label(self, label):
        return self.classes[label]
