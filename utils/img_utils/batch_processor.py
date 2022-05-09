import os
from os.path import join, isfile
from typing import List, Tuple, Iterable

from PIL import Image


class ImageBatchProcessor:
    """
    A class to perform various operations on batch of image paths
    """

    def convert(self, paths_batch: List[str],  mode: str) -> 'ImageBatchProcessor':
        """
        Converts images to specified format and saves them

        :param paths_batch: list of image paths
        :param mode: desired image format
        """

        for path in paths_batch:
            img = Image.open(path).convert(mode=mode)

            os.remove(path)
            img.save(path)

        return self

    def resize(self, paths_batch: List[str], size: Tuple[int, int]) -> 'ImageBatchProcessor':
        """
        Resizes and saves images

        :param paths_batch: list of image paths
        :param size: tuple of images' (new_width, new_height)
        :return: self
        """

        for path in paths_batch:
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
    def filter_invalid_extensions(img_paths: List[str], valid_extension: str) -> List[str]:
        """
        Filters and returns images with invalid extensions

        :param img_paths: list of image paths
        :param valid_extension: valid image extension
        :return: list of image paths with invalid extensions
        """

        invalid_extension_imgs = list(filter(lambda x: not x.endswith(valid_extension), img_paths))

        return invalid_extension_imgs

    @staticmethod
    def load_batch(root: str) -> Iterable[List[str]]:
        """
        Yields a list with all image paths in subfolder

        :param root: path to root folder with images
        :return a list with all image paths
        """

        for name in os.listdir(root):
            folder = join(root, name)
            folder_image_paths = [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))]
            folder_image_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            yield folder_image_paths
