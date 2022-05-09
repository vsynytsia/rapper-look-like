from typing import Tuple, Union

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN


def get_bboxes(img: Union['Image', str], landmarks: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts face and, if specified, face landmarks from image

    :param img: one of: one of: path to image, PIL image object
    :param landmarks: whether to detect face landmarks
    :return: tuple of numpy array with bounding box coordinates and numpy array of face landmarks
    """

    if isinstance(img, str):
        img = Image.open(img).convert('RGB')

    extractor = MTCNN()

    response = extractor.detect(img, landmarks=landmarks)
    bboxes = response[0]
    landmarks_ = response[2][0] if landmarks else None

    return bboxes, landmarks_
