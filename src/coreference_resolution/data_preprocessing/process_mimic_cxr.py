import ast
import json
import logging
import os
import sys
import time

import hydra
from omegaconf import OmegaConf
# pylint: disable=import-error,wrong-import-order
from common_utils.common_utils import check_and_remove_dirs
from coreference_resolution.data_preprocessing import mimic_cxr_csv2conll, mimic_cxr_conll2jsonlines


logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):

    print(OmegaConf.to_yaml(config))

    # The history output dir will be deleted and created again.
    logger.debug("Clean the history files.")
    check_and_remove_dirs(config.output.base_dir, config.clear_history)

    logger.info("*" * 60)
    logger.info("Stage 1: Convert mimic-cxr .csv files to individual .conll files")
    start1 = time.time()
    if os.path.exists(config.temp.base_dir):
        logger.info("Individual .conll files found and will be reused.")
    else:
        log_out = mimic_cxr_csv2conll.prepare_conll(config)
        with open(os.path.join(config.output.run_statistic), "a", encoding="UTF-8") as f:
            f.write(json.dumps(log_out, indent=2))
    stop1 = time.time()

    logger.info("*" * 60)
    logger.info("Stage 2: Aggregrate required conll files for model training.")
    start2 = time.time()
    log_out = mimic_cxr_csv2conll.aggregrate_conll(config)
    with open(os.path.join(config.output.log_file), "w", encoding="UTF-8") as f:
        for split_mode, details in log_out.items():
            f.write(json.dumps({
                "output_folder": split_mode,
                "details": details
            }, indent=2))
            f.write("\n")
    stop2 = time.time()

    logger.info("*" * 60)
    start3 = time.time()
    logger.info("Stage 3: Convert mimic-cxr .conll files to .jsonlines files")
    json_output_dir = mimic_cxr_conll2jsonlines.invoke(config)
    stop3 = time.time()

    # logger.info("*" * 60)
    # # logger.info("CSV format dir: %s", temp_output_dir)
    # # logger.info("Stage 1 - time: %.2fs", stop1-start1)
    # logger.info("CoNLL format dir name: [%s], root dir: %s", config.output.root_dir_name, output_base_dir)
    # logger.info("Stage 2 - time: %.2fs", stop2-start2)
    # logger.info("JSON format dir name: [%s], root dir : %s", os.path.basename(json_output_dir), output_base_dir)
    # logger.info("Stage 3 - time: %.2fs", stop3-start3)
    # logger.info("Total time: %.2fs", stop3-start2)
    # logger.info("*" * 60)


if __name__ == "__main__":
    sys.argv.append("coref_data_preprocessing@_global_=mimic_cxr")
    main()  # pylint: disable=no-value-for-parameter
