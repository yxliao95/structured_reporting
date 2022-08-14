import logging
import os
import time

import hydra
from common_utils.coref_utils import remove_all
from coreference_resolution.data_preprocessing import i2b2_conll2jsonlines
from coreference_resolution.data_preprocessing import i2b2_raw2conll

logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):

    # The history output dir will be deleted and created again.
    config = config.coref_data_preprocessing.i2b2
    if config.clear_history:
        remove_all(config.output_dir)

    logger.info("*" * 60)
    logger.info("Stage 1: Convert i2b2 data to .conll files")
    start1 = time.time()
    conll_output_dir = i2b2_raw2conll.invoke(config)
    stop1 = time.time()

    logger.info("*" * 60)
    logger.info("Stage 1: Convert i2b2 .conll files to .jsonlines files")
    start2 = time.time()
    json_output_dir = i2b2_conll2jsonlines.invoke(conll_output_dir)
    stop2 = time.time()

    logger.info("*" * 60)
    logger.info("CoNLL format dir: %s", conll_output_dir)
    logger.info("Stage 1 - time: %.2fs", stop1-start1)
    logger.info("JSON format dir: %s", json_output_dir)
    logger.info("Stage 2 - time: %.2fs", stop2-start2)
    logger.info("Total time: %.2fs", stop2-start1)
    logger.info("*" * 60)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
