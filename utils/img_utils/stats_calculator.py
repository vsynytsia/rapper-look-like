from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms as T

from .load_image_paths import load_dataset_paths


class ImageStatsCalculator:
    """
    Class to aggregate various statistical features of image dataset(folder)
    """

    def __init__(self, root: str) -> None:
        """
        :param root: path to root folder
        """

        self.root = root
        self.mean, self.std = None, None

    def calculate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes mean and standard deviation over whole dataset

        :return: tuple of torch tensors [c1_mean, c2_mean, c3_mean], [c1_std, c2_std, c3_std]
         where c1, c2, c3 refer to color channels
        """

        all_img_paths = load_dataset_paths(self.root)

        if (self.mean and self.std) is None:
            self.mean, self.std = 0, 0

        for img_path in all_img_paths:
            img_matrix = self._prepare_img(img_path)
            self.mean += img_matrix.mean(1)
            self.std += img_matrix.std(1)

        self.mean /= len(all_img_paths)
        self.std /= len(all_img_paths)

        return self.mean, self.std

    def _prepare_img(self, path: str) -> torch.Tensor:
        """
        Converts an image to pytorch tensor suitable for further statistical features aggregation

        :param path: path ot image
        :return: image tensor
        """

        img_tensor = self._img_to_tensor(path)
        prepared_tensor = self._prepare_img_tensor(img_tensor)

        return prepared_tensor

    @staticmethod
    def _img_to_tensor(path: str) -> torch.Tensor:
        """
        Opens an image, converts it to RGB format and then converts it to pytorch tensor

        :param path: path to image
        :return: tensor of shape (C x H x W)
        """

        img = Image.open(path)
        img_tensor = T.ToTensor()(img)

        return img_tensor

    @staticmethod
    def _prepare_img_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Resizes tensor so statistical features can be computed

        :param img_tensor: tensor of shape (C x H x W)
        :return: tensor of shape (C x (H * W))
        """

        prepared_tensor = img_tensor.view(img_tensor.size(0), -1)

        return prepared_tensor
