import os
from typing import List, Tuple

import yaml

from utils.img_utils import ImageBatchProcessor
from utils.logger import get_logger
from .cleaner import DatasetCleaner
from .fetcher import DatasetFetcher

config = yaml.safe_load(open('config/config.yaml'))
logger = get_logger(config['logger']['app_name'], __name__)


def _update_config(path: str, new_names: List[str]) -> None:
    """
    Updates config file by writing new rapper names to it

    :param path: path to config file
    :param new_names: list of new rapper names
    """

    data = yaml.safe_load(open(path))
    duplicates = list(set(data['images']['labels']) & set(new_names))
    if len(duplicates) > 0:
        raise ValueError(f'Config file already contains {duplicates}')

    data['images']['labels'].extend(new_names)
    with open(path, 'w') as cfg_file:
        cfg_file.write(yaml.dump(data, default_flow_style=False, sort_keys=False))


def extend_dataset(
        new_names: List[str],
        output_folder: str,
        img_number: int,
        clean: bool = False,
        img_size: Tuple[int, int] = None,
        valid_format: str = None,
        valid_extension: str = None
        ) -> None:
    """
    Extends existing dataset by adding new images(specified by parameter new_names)

    :param new_names: list of new rapper names images of whom will be downloaded
    :param output_folder: path to output folder with images
    :param img_number: number of images per folder
    :param clean: whether to clean extended dataset
    :param img_size: tuple of image width and height
    :param valid_format: valid image format(RGB, RGBA etc.)
    :param valid_extension: valid image extension(JPG, PNG etc.)
    """

    _update_config(path='config/config.yaml', new_names=new_names)
    logger.info('Updated config file with new names')

    fetcher = DatasetFetcher(
        names=new_names,
        output_folder=output_folder,
        img_limit=img_number,
        delete_existing=False
        )

    logger.info('Started extending dataset')
    fetcher.fetch_dataset()
    logger.info('Finished extending dataset')

    if clean:
        cleaner = DatasetCleaner(
            root=output_folder,
            img_size=img_size,
            valid_extension=valid_extension,
            valid_format=valid_format
            )

        folder_paths = list(map(lambda x: os.path.join(config['images']['root'], x), new_names))
        batch_processor = ImageBatchProcessor()

        logger.info('Started cleaning extended dataset')
        for i, folder_path in enumerate(folder_paths):
            logger.info(f'Started cleaning {new_names[i]} folder')
            invalid_imgs = cleaner.clean_folder(folder_path)
            batch_processor.delete(invalid_imgs)
            logger.info(f'Deleted {len(invalid_imgs)} images from {new_names[i]} folder')
        logger.info('Finished cleaning extended dataset')
