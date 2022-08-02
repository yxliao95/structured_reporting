import pandas
import os, sys
from os import path
import hydra
import logging

logger = logging.getLogger()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    logger.debug(config)


if __name__ == "__main__":
    main()
