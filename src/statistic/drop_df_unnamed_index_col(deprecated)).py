import logging
import os
from tqdm import tqdm

import hydra
import pandas as pd
from omegaconf import OmegaConf
# pylint: disable=import-error
from common_utils.file_checker import FileChecker

logger = logging.getLogger()

module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")
FILE_CHECKER = FileChecker()


@hydra.main(version_base=None, config_path=config_path, config_name="nlp_ensemble")
def main(config):
    print(OmegaConf.to_yaml(config))

    sections = [_sectionName for _sectionName in config.output.section.keys()]

    for section_name in tqdm(sections):
        # CSV files base dir for each sections
        csv_file_dir = os.path.join(config.output.dir, section_name)
        for file_entry in tqdm(os.scandir(csv_file_dir)):
            if FILE_CHECKER.ignore(file_entry.name):
                continue
            # Read csv
            df_base = pd.read_csv(file_entry.path, index_col=0)
            # Drop extra index columns
            df_base.drop(df_base.filter(regex='Unnamed').columns, axis=1, inplace=True)
            # Write csv
            df_base.to_csv(file_entry.path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
