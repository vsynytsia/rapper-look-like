import pickle
import warnings

import yaml

from data.dataset import create_dataset
from tools import filter_inference_images
from tools import train, inference
from utils import plot
from utils.logger import setup_logger

warnings.filterwarnings("ignore")

config = yaml.safe_load(open('config/config.yaml'))
logger = setup_logger(config['logger']['app_name'])


def main():
    if config['images']['download']:
        logger.info('Started creating dataset')
        create_dataset(
            names=config['images']['labels'],
            output_folder=config['images']['root'],
            img_limit=100,
            delete_existing=True,
            clean=True,
            img_size=(config['images']['width'], config['images']['height']),
            valid_format=config['images']['mode'],
            valid_extension=config['images']['extension']
            )
        logger.info('Finished creating dataset')

    if config['train']['do_train']:
        logger.info('Started training')
        embeddings = None

        if config['train']['load_embeddings']:
            with open(config['embeddings']['path'], 'rb') as input_file:
                embeddings = pickle.load(input_file)
            logger.info('Embeddings loaded successfully')
            train(root=config['images']['root'], embeddings=embeddings)

        else:
            logger.info('Started creating image embeddings')
            train(root=config['images']['root'])
            logger.info('Finished creating image embeddings')

    input_image_folder = config['inference']['images_folder']
    valid_input_imgs = filter_inference_images(input_image_folder)

    logger.info(f'Got {len(valid_input_imgs)} images as input. Started inference')
    pred_imgs, labels = inference(valid_input_imgs)
    logger.info('Finished inference')

    logger.info('Plotting results')
    plot(valid_input_imgs, pred_imgs, labels)
    logger.info('Results saved in a file result.png')


if __name__ == '__main__':
    main()
