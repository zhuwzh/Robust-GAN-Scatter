# -*- coding: utf-8 -*-
import os
import logging
from fgan import fgan
from config import get_config

logger = logging.getLogger('InfoLog')

def main(config):
    logger.info(f'***START TRAINING ***'.upper())
    logger.info(config.__dict__)
    f = fgan(config)
    f.train()

if __name__ == '__main__':
    config = get_config()
    main(config)