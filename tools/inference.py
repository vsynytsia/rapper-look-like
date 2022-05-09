import pickle
from typing import List, Tuple

import numpy as np
import yaml

from data.dataset import DatasetCleaner
from embeddings import get_embedding
from face_processing import extract_face
from utils.img_utils import load_dataset_paths, load_folder_paths
from utils.logger import get_logger

config = yaml.safe_load(open('config/config.yaml'))
logger = get_logger(config['logger']['app_name'], __name__)


def _nearest_indices_to_images(indices: np.ndarray) -> List[str]:
    """
    Returns images of corresponding nearest neighbors indices

    :param indices: numpy array with neighbors indices
    :return: list of image paths
    """

    nearest_neighbors = []
    all_paths = load_dataset_paths(config['images']['root'])

    for index in indices.flatten():
        nearest_neighbors.append(all_paths[index])

    return nearest_neighbors


def filter_inference_images(folder_path: str) -> List[str]:
    """
    Preprocesses inference images:
    - converts all images to one format;
    - resizes all images;
    - changes all files extensions;
    - finds images where 0 or more than 1 people present;
    - finds duplicate or very similar images

    :param folder_path: path to folder with inference images
    :return: list of valid inference images
    """

    cleaner = DatasetCleaner(
        root=folder_path,
        img_size=(config['images']['width'], config['images']['height']),
        valid_extension=config['images']['extension'],
        valid_format=config['images']['mode']
    )

    all_input_imgs = load_folder_paths(folder_path)
    invalid_input_imgs = cleaner.clean_folder(folder_path)

    if len(invalid_input_imgs) != 0:
        logger.warn(f'Detected {len(invalid_input_imgs)} images with either 0 or more than 1 faces on them:'
                    f'{invalid_input_imgs}. They will be ignored')

    valid_input_imgs = list(set(all_input_imgs) - set(invalid_input_imgs))
    return valid_input_imgs


def inference(img_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Predicts most similar rapper image and its class label

    :param img_paths: list of input image paths
    :return: list of tuples for each image: (most_similar_image_path, class_label)
    """

    with open(config['model']['clf_path'], 'rb') as input_file:
        image_clf = pickle.load(input_file)
        logger.info('Classifier loaded successfully')

    face_imgs, invalid_imgs = extract_face(img_paths)
    embeddings = get_embedding(face_imgs)

    if len(invalid_imgs) != 0:
        logger.warn(f"Couldn't extract faces from {len(invalid_imgs)} images: {invalid_imgs}. They will be ignored")

    nearest_neighbor_indices = image_clf.kneighbors(embeddings, 1, return_distance=False)
    nearest_neighbors = _nearest_indices_to_images(nearest_neighbor_indices)
    classes_pred = list(map(lambda x: x.split('/')[2], nearest_neighbors))

    return nearest_neighbors, classes_pred
