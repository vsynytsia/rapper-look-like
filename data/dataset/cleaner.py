import os
from typing import Tuple, List

import yaml

from face_processing import handle_face_number
from utils import logger
from utils.img_utils import DuplicatesHandler
from utils.img_utils import ImageBatchProcessor
from utils.img_utils import load_folder_paths

config = yaml.safe_load(open('config/config.yaml'))
logger = logger.get_logger(config['logger']['app_name'], __name__)


class DatasetCleaner:
    """
    Class to clean downloaded dataset
    """

    def __init__(
            self,
            root: str,
            img_size: Tuple[int, int],
            valid_extension: str,
            valid_format: str
            ) -> None:
        """
        :param root: path to root folder
        :param img_size: tuple of image's (new_width, new_height)
        :param valid_extension: valid image extension
        :param valid_format: valid image format
        """

        self.root = root
        self.img_size = img_size
        self.valid_extension = valid_extension
        self.valid_format = valid_format

    def clean_folder(self, folder_path: str) -> List[str]:
        """
        Cleans single image folder:
        - convert all images to one format;
        - resize all images;
        - change all files extensions;
        - find images where 0 or more than 1 people present;
        - find duplicate or very similar images

        :param folder_path: path to folder
        :return: list of invalid images
        """

        duplicates_handler = DuplicatesHandler(similarity=90)
        batch_processor = ImageBatchProcessor()

        img_paths_batch = load_folder_paths(folder_path)
        batch_processor.convert(img_paths_batch, mode=self.valid_format).resize(img_paths_batch, size=self.img_size)

        invalid_imgs = []
        invalid_imgs.extend(handle_face_number(img_paths_batch))
        invalid_imgs.extend(duplicates_handler.handle(img_paths_batch))

        invalid_imgs = list(set(invalid_imgs))
        valid_img_paths = list(set(img_paths_batch) - set(invalid_imgs))

        invalid_extension_imgs = batch_processor.filter_invalid_extensions(valid_img_paths,
                                                                           config['images']['extension'])
        batch_processor.change_extension(invalid_extension_imgs, self.valid_extension)
        return invalid_imgs

    def clean_dataset(self) -> None:
        """
        Cleans entire image dataset:
        - convert all images to one format;
        - resize all images;
        - change all files extensions;
        - delete images where 0 or more than 1 people present;
        - remove duplicate or very similar images
        """

        root = config['images']['root']
        batch_processor = ImageBatchProcessor()

        for label in config['images']['labels']:
            folder_path = os.path.join(root, label)

            logger.info(f'Started cleaning {label} folder')
            invalid_imgs = self.clean_folder(folder_path)
            batch_processor.delete(invalid_imgs)
            logger.info(f'Deleted {len(invalid_imgs)} images from {label} folder')
