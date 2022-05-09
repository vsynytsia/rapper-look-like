import pickle

import numpy as np
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from embeddings import get_embedding
from face_processing import extract_face
from utils.img_utils import ImageBatchProcessor
from utils.img_utils import load_dataset_paths
from utils.logger import get_logger

config = yaml.safe_load(open('config/config.yaml'))
logger = get_logger(config['logger']['app_name'], __name__)


def train(root: str, embeddings: np.ndarray = None) -> None:
    """
    Trains a classifier

    :param root: path to root folder
    :param embeddings: optional, numpy array of embeddings
    """

    all_img_paths = load_dataset_paths(root)
    all_img_labels = list(map(lambda x: x.split('/')[2], all_img_paths))

    if embeddings is None:
        logger.info('Started extracting faces from images')
        face_imgs, invalid_imgs = extract_face(all_img_paths)
        logger.info('Finished extracting faces from images')

        if len(invalid_imgs) != 0:
            batch_processor = ImageBatchProcessor()
            batch_processor.delete(invalid_imgs)
            logger.info(f'Couldnt extract faces from {len(invalid_imgs)} photos. Deleted them.')

        logger.info('Started creating face embeddings')
        embeddings = get_embedding(face_imgs)
        logger.info('Finished creating face embeddings')

        with open(config['embeddings']['path'], 'wb') as output_file:
            pickle.dump(embeddings, output_file)
            logger.info('Embeddings saved successfully')

    encoder = LabelEncoder()
    encoder.fit(config['images']['labels'])
    labels_encoded = encoder.transform(all_img_labels)

    logger.info('Started fitting classifier')
    images_clf = KNeighborsClassifier(n_neighbors=1)
    images_clf.fit(embeddings, labels_encoded)
    logger.info('Finished fitting classifier')

    with open(config['model']['clf_path'], 'wb') as output_file:
        pickle.dump(images_clf, output_file)
        logger.info('Classifier saved successfully')

    with open(config['model']['encoder_path'], 'wb') as output_file:
        pickle.dump(encoder, output_file)
