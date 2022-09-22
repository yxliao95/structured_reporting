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
from common_utils.common_utils import check_and_remove_dirs
from common_utils.file_checker import FileChecker

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")

FILE_CHECKER = FileChecker()


@hydra.main(version_base=None, config_path=config_path, config_name="statistic")
def main(config):
    print(OmegaConf.to_yaml(config))
    input_cfg = config.input

    check_and_remove_dirs(config.output_dir, config.clear_history)

    # Loop in sections
    section_name_list = input_cfg.section
    for section_name in section_name_list:
        logger.info("Processing section: %s", section_name)
        for target_dir in input_cfg.target_dir:
            target_section_dir = os.path.join(target_dir, section_name)
            for entry in os.scandir(target_section_dir):
                if FILE_CHECKER.ignore(entry.path):
                    break
                df = pd.read_csv(entry.path, index_col=0)
                df_hasDuplicate = df[df.filter(like="align_to_spacy").duplicated(keep=False)]
                if not df_hasDuplicate.empty:
                    print(entry.path)
                    print(df_hasDuplicate)


if __name__ == "__main__":
    sys.argv.append("+statistic/alignment@_global_=i2b2")
    main()  # pylint: disable=no-value-for-parameter
