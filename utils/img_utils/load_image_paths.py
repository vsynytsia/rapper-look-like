import os
from typing import List


def load_folder_paths(root: str) -> List[str]:
    """
    Creates a list of full paths to images in given folder

    :param root: folder path
    :return: list of full image paths
    """

    folder_img_paths = [os.path.join(root, f).replace('\\', '/')
                        for f in os.listdir(root)
                        if os.path.isfile(os.path.join(root, f))]

    return folder_img_paths


def load_dataset_paths(root: str) -> List[str]:
    """
    Creates a list of full paths to all images in dataset folder

    :param root: path to root folder
    :return: a list with all image paths
    """

    all_img_paths = []
    for name in os.listdir(root):
        folder_path = os.path.join(root, name)
        folder_img_paths = load_folder_paths(folder_path)
        all_img_paths.extend(folder_img_paths)

    return all_img_paths
