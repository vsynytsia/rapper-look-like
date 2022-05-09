import logging
import sys


def setup_logger(logger_name, filename=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(sh)

    if filename:
        fh = logging.FileHandler(filename=filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(app_name, module_name):
    return logging.getLogger(app_name).getChild(module_name)
