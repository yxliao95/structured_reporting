import json
import logging
import os
import sys
import time
import re
import ast

import hydra
import pandas as pd
from omegaconf import OmegaConf

# pylint: disable=import-error,wrong-import-order
from common_utils.file_checker import FileChecker

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

FILE_CHECKER = FileChecker()


@hydra.main(version_base=None, config_path=config_path, config_name="statistic")
def main(config):
    print(OmegaConf.to_yaml(config))

    # Read spacy output as baseline

    # Read 3 coref model's output

    # Align to spacy


if __name__ == "__main__":
    sys.argv.append("statistic@_global_=coref_voting")
    main()  # pylint: disable=no-value-for-parameter
