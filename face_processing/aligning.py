import numpy as np
from PIL import Image

from .detection import get_bboxes


def align_face(path: str) -> Image:
    """
    Rotates on image so that eyes are located on a horizontal line

    :param path: path to image
    :return: aligned PIL image
    """

    img = Image.open(path)

    _, landmarks = get_bboxes(img, landmarks=True)
    right_eye, left_eye = landmarks[0], landmarks[1]

    x1, y1 = right_eye
    x2, y2 = left_eye

    a, b = y1 - y2, x2 - x1
    c = np.sqrt(a**2 + b**2)

    cos_alpha = (b**2 + c**2 - a**2) / (2 * b * c)
    alpha = np.rad2deg(np.arccos(cos_alpha))

    if y1 < y2:
        aligned_img = img.rotate(alpha)
    else:
        aligned_img = img.rotate(360 - alpha)

    return aligned_img
