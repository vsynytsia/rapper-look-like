from typing import List

import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm


def get_embedding(imgs: List['Image']) -> np.ndarray:
    """
    Creates a 512-dimensional representation of given image

    :param imgs: list of PIL Images
    :return: numpy array of 512-dimensional representations of given images
    """

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    embeddings = np.empty(shape=(1, 512))

    for img in tqdm(imgs):
        img_tensor = ToTensor()(img)
        img_embedding = resnet(img_tensor.unsqueeze(0)).data
        embeddings = np.vstack((embeddings, img_embedding))

    return embeddings[1:]
