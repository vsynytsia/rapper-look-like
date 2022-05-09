from itertools import combinations
from typing import List

import imagehash
import numpy as np
from PIL import Image


class DuplicatesHandler:
    """
    Class to handle duplicate or similar images
    """

    def __init__(self, similarity: int, hash_size: int = 8) -> None:
        """
        :param similarity: minimal 'similarity' of 2 images to be considered duplicates. similarity є [0, 100]
        :param hash_size:
        """

        self.similarity = similarity
        self.hash_size = hash_size

    def handle(self, paths: List[str]) -> List[str]:
        """
        Finds duplicates or similar images in the list

        :param paths: list of image paths
        :return: list of paths to similar images
        """

        similar_and_duplicates = []

        for path1, path2 in combinations(paths, 2):

            if self.are_similar(path1, path2) or self.are_duplicates(path1, path2):
                similar_and_duplicates.append(path1)

        return similar_and_duplicates

    @staticmethod
    def are_duplicates(path1: str, path2: str) -> bool:
        """
        Checks if two images are exact duplicates

        :param path1: path to first image
        :param path2: path to second image
        :return: True if two images are exact duplicates, False otherwise
        """

        img1, img2 = Image.open(path1), Image.open(path2)

        return list(img1.getdata()) == list(img2.getdata())

    def are_similar(self, path1: str, path2: str) -> bool:
        """
        Checks if two images are similar by computing Hamming distance

        :param path1: first image
        :param path2: second image
        :return: True if  Hamming distance between images <= diff_limit, False otherwise
        """

        threshold = 1 - self.similarity / 100
        diff_limit = int(threshold * self.hash_size**2)

        img1, img2 = Image.open(path1), Image.open(path2)

        hash1 = imagehash.average_hash(img1, self.hash_size).hash
        hash2 = imagehash.average_hash(img2, self.hash_size).hash

        similarity = np.count_nonzero(hash1 != hash2) <= diff_limit

        return similarity
