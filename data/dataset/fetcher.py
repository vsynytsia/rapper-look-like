import os
import shutil
from pathlib import Path
from typing import List

import yaml
from bing_image_downloader.downloader import download

from utils import logger

config = yaml.safe_load(open('config/config.yaml'))
logger = logger.get_logger(config['logger']['app_name'], __name__)


class DatasetFetcher:
    """
    Class to download rappers face images using Bing searcher
    """

    def __init__(
            self,
            names: List[str],
            output_folder: str,
            img_limit: int,
            delete_existing: bool
            ) -> None:
        """
        :param names: list of rappers images of whom are to be downloaded
        :param output_folder: name of the root folder
        :param img_limit: number of images to download for each rapper
        :param delete_existing: whether to delete existing output_folder
        """

        self.names = names
        self.output_folder = output_folder
        self.img_limit = img_limit
        self.delete_existing = delete_existing

    def fetch_dataset(self) -> None:
        """
        Downloads images of rappers specified in the config file and saves them to output_folder
        """

        if self.delete_existing:
            self.delete_existing_dataset_folder()

        for name in self.names:
            self.download_batch(name)
            logger.debug(f'Successfully fetched {name} images')

    def download_batch(self, name: str) -> None:
        """
        Downloads batch of images

        :param name: name of the rapper images of whom are to be downloaded
        """

        query = f'{name} rapper face'
        download(
            query=query,
            limit=self.img_limit,
            adult_filter_off=False,
            output_dir=self.output_folder,
            filter='photo',
            timeout=10,
            verbose=False
        )

        # downloaded folder's name is '{name} rapper face'
        # so to rename it back to {name} we need to extract name from the query
        folder_name = " ".join(query.split()[:-2])
        old_folder_path = os.path.join(self.output_folder, query)
        new_folder_path = os.path.join(self.output_folder, folder_name)
        os.rename(old_folder_path, new_folder_path)

    def delete_existing_dataset_folder(self) -> None:
        """
        Checks if dataset(output_folder) folder already exists. If so, deletes it
        """

        folder_path = Path(self.output_folder)
        logger.debug('Checking if images folder exists')

        if folder_path.exists() and folder_path.is_dir():
            logger.debug('Existing images folder found. Deleting it')
            shutil.rmtree(folder_path)
        else:
            logger.debug('Existing images folder not found. Creating it')
