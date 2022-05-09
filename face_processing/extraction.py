from typing import List, Tuple

from PIL import Image

from .aligning import align_face
from .detection import get_bboxes


def extract_face(paths: List[str]) -> Tuple[List['Image'], List[str]]:
    """
    Extracts face from the image where only 1 person present

    :param paths: list of image paths
    :return: list of PIL face images and list of invalid images(images from which neural network couldn't extract faces)
    """

    faces, invalid_imgs = [], []

    for path in paths:
        aligned_img = align_face(path)

        try:
            bbox = get_bboxes(aligned_img)[0][0]
        except TypeError:
            invalid_imgs.append(path)
            continue

        face = aligned_img.crop(bbox).resize((160, 160))

        try:
            faces.append(face)
        except ValueError:
            invalid_imgs.append(path)

    return faces, invalid_imgs
