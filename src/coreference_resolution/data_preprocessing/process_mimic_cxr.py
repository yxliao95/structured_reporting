import logging
import os
import time

import hydra
from common_utils.coref_utils import remove_all
from coreference_resolution.data_preprocessing import mimic_cxr_csv2conll, mimic_cxr_raw2csv, mimic_cxr_conll2jsonlines


logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):

    # The history output dir will be deleted and created again.
    mimic_cfg = config.coref_data_preprocessing.mimic_cxr
    if mimic_cfg.clear_history and os.path.exists(mimic_cfg.output_dir):
        if mimic_cfg.remove_temp:
            remove_all(mimic_cfg.output_dir)
        else:
            for entry in os.scandir(mimic_cfg.output_dir):
                if entry.path != mimic_cfg.temp_dir:
                    remove_all(entry.path)

    logger.info("*" * 60)
    logger.info("Stage 1: Preprocess mimic-cxr data with SpaCy and CoreNLP.")
    start1 = time.time()
    if not mimic_cfg.reload_from_temp:
        temp_output_dir = mimic_cxr_raw2csv.run(config)
    else:
        logger.info("Skipped")
        temp_output_dir = mimic_cfg.temp_dir
    stop1 = time.time()

    logger.info("*" * 60)
    logger.info("Stage 2: Convert mimic-cxr .csv files to .conll files")
    start2 = time.time()
    output_base_dir = mimic_cxr_csv2conll.invoke(config, temp_output_dir)
    stop2 = time.time()

    logger.info("*" * 60)
    start3 = time.time()
    logger.info("Stage 3: Convert mimic-cxr .conll files to .jsonlines files")
    json_output_dir = mimic_cxr_conll2jsonlines.invoke(config, output_base_dir)
    stop3 = time.time()

    logger.info("*" * 60)
    logger.info("CSV format dir: %s", temp_output_dir)
    logger.info("Stage 1 - time: %.2fs", stop1-start1)
    logger.info("CoNLL format dir name: [%s], root dir: %s", mimic_cfg.output.root_dir_name, output_base_dir)
    logger.info("Stage 2 - time: %.2fs", stop2-start2)
    logger.info("JSON format dir name: [%s], root dir : %s", os.path.basename(json_output_dir), output_base_dir)
    logger.info("Stage 3 - time: %.2fs", stop3-start3)
    logger.info("Total time: %.2fs", stop3-start1)
    logger.info("*" * 60)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
