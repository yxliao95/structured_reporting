import os
import sys
import time
import logging

import hydra
from omegaconf import OmegaConf

# pylint: disable=import-error,wrong-import-order
from coreference_resolution.data_preprocessing import i2b2_raw2conll
from coreference_resolution.data_preprocessing import i2b2_conll2jsonlines
from common_utils.common_utils import check_and_remove_dirs

logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):
    print(OmegaConf.to_yaml(config))

    # The history output dir will be deleted and created again.
    check_and_remove_dirs(config.output.base_dir, config.clear_history)

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
    sys.argv.append("+coreference_resolution/data_preprocessing@_global_=i2b2")
    main()  # pylint: disable=no-value-for-parameter
