import os

import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from datasets.base_dataset import BaseDataset
from utils.augmenters.augment import seg


EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class FER2013Dataset(BaseDataset):
    """
    Input params:
        stage: The stage of training.
        configuration: Configuration dictionary.
    """
    def __init__(self, configuration):
        super().__init__(configuration)

        self._stage = configuration["stage"]

        self._image_size = tuple(configuration["input_size"])

        self._data = pd.read_csv(os.path.join(configuration["dataset_path"], "{}.csv".format(self._stage)))

        self._pixels = self._data["pixels"].tolist()
        self._emotions = pd.get_dummies(self._data["emotion"])

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )


    def __getitem__(self, index):
        pixels = self._pixels[index]
        pixels = list(map(int, pixels.split(" ")))
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)

        # print(self._image_size)
        image = cv2.resize(image, self._image_size)

        image = np.dstack([image] * 1)
        # image = np.dstack([image] * 3)

        # if self._stage == "train":
        image = seg(image=image)

        # if self._stage == "test" and self._tta == True:
        #     images = [seg(image=image) for i in range(self._tta_size)]
        #     # images = [image for i in range(self._tta_size)]
        #     images = list(map(self._transform, images))
        #     target = self._emotions.iloc[idx].idxmax()
        #     return images, target

        image = self._transform(image)
        target = self._emotions.iloc[index].idxmax()
        return image, target

    def __len__(self):
        # return the size of the dataset
        return len(self._pixels)
