from typing import List

from .extraction import get_bboxes


def num_faces_on_image(path: str) -> int:
    """
    Detects number of faces(bounding boxes) on the image

    :param path: path to image
    :return: number of detected images(bounding boxes)
    """

    bboxes = get_bboxes(path)[0]

    return len(bboxes) if bboxes is not None else 0


def handle_face_number(paths: List[str], faces_allowed: int = 1) -> List[str]:
    """
    Finds and images with 0 or more than 1 faces present

    :param paths: list of images paths
    :param faces_allowed: number of faces allowed on the photo
    :return: list of paths to images with 0 or more than 1 faces present
    """

    invalid_imgs = []

    for path in paths:
        if num_faces_on_image(path) != faces_allowed:
            invalid_imgs.append(path)

    return invalid_imgs
