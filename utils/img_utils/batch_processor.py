import os
from typing import List, Tuple, Iterable

from PIL import Image

from .load_image_paths import load_folder_paths


class ImageBatchProcessor:
    """
    A class to perform various operations on image paths batch
    """

    def convert(self, paths: List[str],  mode: str) -> 'ImageBatchProcessor':
        """
        Converts images to specified format and saves them

        :param paths: list of image paths
        :param mode: desired image format
        """

        for path in paths:
            img = Image.open(path).convert(mode=mode)

            os.remove(path)
            img.save(path)

        return self

    def resize(self, paths: List[str], size: Tuple[int, int]) -> 'ImageBatchProcessor':
        """
        Resizes and saves images

        :param paths: list of image paths
        :param size: tuple of images' (new_width, new_height)
        :return: self
        """

        for path in paths:
            img = Image.open(path).resize(size)

            os.remove(path)
            img.save(path)

        return self

    def change_extension(self, paths: List[str], new_ext: str) -> 'ImageBatchProcessor':
        """
        Changes all files extension and saves them

        :param paths: list of image paths
        :param new_ext: new image extension
        :return: self
        """

        for path in paths:
            name, extension = os.path.splitext(path)
            Image.open(path).convert('RGB').save(f'{name}.{new_ext}')
            os.remove(path)

        return self

    def delete(self, paths: List[str]) -> 'ImageBatchProcessor':
        """
        Deletes batch of images

        :param paths: list with image paths
        :return: self
        """

        for path in paths:
            os.remove(path)

        return self

    @staticmethod
    def filter_invalid_extensions(paths: List[str], valid_extension: str) -> List[str]:
        """
        Filters and returns images with invalid extensions

        :param paths: list of image paths
        :param valid_extension: valid image extension
        :return: list of image paths with invalid extensions
        """

        invalid_extension_imgs = list(filter(lambda x: not x.endswith(valid_extension), paths))

        return invalid_extension_imgs

    @staticmethod
    def load_batch(root: str) -> Iterable[List[str]]:
        """
        Yields a list with all image paths in folder

        :param root: path to root folder with images
        :return a list with all image paths
        """

        for name in os.listdir(root):
            folder_path = os.path.join(root, name)
            folder_image_paths = load_folder_paths(folder_path)
            folder_image_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            yield folder_image_paths
