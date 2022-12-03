import json
import logging
import os
import sys

import hydra
from omegaconf import OmegaConf
# pylint: disable=import-error,wrong-import-order
from common_utils.common_utils import check_and_remove_dirs, check_and_remove_file
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
    logger.info("Stage 1-1: Convert all mimic-cxr .csv files to individual .conll files")
    if os.path.exists(config.temp_pred.base_dir):
        logger.info("Individual conll files found and will be reused.")
    else:
        log_out = mimic_cxr_csv2conll.prepare_conll(config, config.input_pred, config.temp_pred)
        with open(config.output.run_statistic, "a", encoding="UTF-8") as f:
            f.write(f"Source: {config.temp_pred.base_dir} \n")
            f.write(json.dumps(log_out, indent=2))
            f.write("\n")

    logger.info("*" * 60)
    logger.info("Stage 1-2: Convert mimic-cxr .csv files (manual annotated testset) to individual .conll files")
    check_and_remove_dirs(config.temp_gt.base_dir, config.temp_gt.force_run)
    if os.path.exists(config.temp_gt.base_dir):
        logger.info("Individual test conll files found and will be reused.")
    else:
        log_out = mimic_cxr_csv2conll.prepare_conll(config, config.input_gt, config.temp_gt)
        with open(config.output.run_statistic, "a", encoding="UTF-8") as f:
            f.write(f"Source: {config.temp_gt.base_dir} \n")
            f.write(json.dumps(log_out, indent=2))
            f.write("\n")

    logger.info("*" * 60)
    logger.info("Stage 2: Aggregrate required conll files for model training.")
    log_out = mimic_cxr_csv2conll.aggregrate_conll(config)
    with open(config.output.log_file, "a", encoding="UTF-8") as f:
        for split_mode, details in log_out.items():
            f.write(json.dumps({
                "output_folder": split_mode,
                "details": details
            }, indent=2))
            f.write("\n")

    logger.info("*" * 60)
    logger.info("Stage 3: Convert mimic-cxr .conll files to .jsonlines files")
    mimic_cxr_conll2jsonlines.invoke(config)
    logger.info("Done.")


if __name__ == "__main__":
    sys.argv.append("+coreference_resolution/data_preprocessing@_global_=mimic_cxr")
    main()  # pylint: disable=no-value-for-parameter
