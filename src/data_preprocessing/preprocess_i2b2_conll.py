###
# This pre-processing script aims to aggregrate the i2b2 conll files for later usage
###
import logging
import os
import sys
import time
import json
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf
# pylint: disable=import-error
from common_utils.file_checker import FileChecker


logger = logging.getLogger()
FILE_CHECKER = FileChecker()

module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")


@hydra.main(version_base=None, config_path=config_path, config_name="data_preprocessing")
def main(config):
    """ This pre-processing script aims to aggregrate the i2b2 conll files for later usage. """
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    sys.argv.append("data_preprocessing@_global_=i2b2")
    main()  # pylint: disable=no-value-for-parameter
