from typing import List, Tuple

import yaml

from utils import logger
from .cleaner import DatasetCleaner
from .fetcher import DatasetFetcher

config = yaml.safe_load(open('config/config.yaml'))
logger = logger.setup_logger(config['logger']['app_name'])


def create_dataset(
        names: List[str],
        output_folder: str,
        img_number: int,
        delete_existing: bool,
        clean: bool = False,
        img_size: Tuple[int, int] = None,
        valid_format: str = None,
        valid_extension: str = None
        ) -> None:
    """
    Creates and cleans(if needed) image dataset

    :param names: list of rapper names images of whom will be downloaded
    :param output_folder: path to output folder with images
    :param img_number: number of images per folder
    :param delete_existing: whether delete already existing dataset folder
    :param clean: optional, whether to clean downloaded dataset
    :param img_size: optional, tuple of image width and height
    :param valid_format: optional, valid image format(RGB, RGBA etc.)
    :param valid_extension: optional, valid image extension(JPG, PNG etc.)
    """

    fetcher = DatasetFetcher(
        names=names,
        output_folder=output_folder,
        img_limit=img_number,
        delete_existing=delete_existing
        )

    logger.info('Started fetching dataset')
    fetcher.fetch_dataset()
    logger.info('Finished fetching dataset')

    if clean:
        cleaner = DatasetCleaner(
            root=output_folder,
            img_size=img_size,
            valid_extension=valid_extension,
            valid_format=valid_format
            )

        logger.info('Started cleaning dataset')
        cleaner.clean_dataset()
        logger.info('Finished cleaning dataset')
