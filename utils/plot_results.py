from typing import List

import matplotlib.pyplot as plt
from PIL import Image


def plot(
        input_image_paths: List[str],
        pred_image_paths: List[str],
        labels: List[str],
        ) -> None:
    """
    Plots pairs of input images and predicted images and saves the plot

    :param input_image_paths: list of paths to inference images
    :param pred_image_paths: list of paths to predicted images
    :param labels: list of predicted labels
    """

    fig, axs = plt.subplots(nrows=len(input_image_paths), ncols=2, figsize=(15, 15))

    for i, (input_path, pred_path, label) in enumerate(zip(input_image_paths, pred_image_paths, labels)):
        axs[i, 0].imshow(Image.open(input_path))
        axs[i, 1].imshow(Image.open(pred_path))

        axs[i, 0].get_xaxis().set_visible(False)
        axs[i, 0].get_yaxis().set_visible(False)
        axs[i, 1].get_xaxis().set_visible(False)
        axs[i, 1].get_yaxis().set_visible(False)

        axs[i, 0].set_title(f'You look like {label}!')
        axs[i, 1].set_title(f'{label}')

    fig.tight_layout()
    plt.savefig('result.png')
