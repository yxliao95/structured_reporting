import i2b2_raw2conll as i2b2_raw2conll
import i2b2_conll2jsonlines as i2b2_conll2jsonlines
import logging, os, hydra, time

logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):
    src = os.path.join(coref_module_path, "fast-coref", "src")
    if not os.path.exists(src):
        logger.error(
            f"Directory 'fast-coref/' not found. Please clone the 'fast-coref' repo to {coref_module_path}, e.g.: git clone git@github.com:liaoooyx/fast-coref.git"
        )
        raise Exception(f"Lacking the fast-coref repo in {coref_module_path}")
    config = config.coref_data_preprocessing.i2b2
    start1 = time.time()
    conll_output_dir = i2b2_raw2conll.invoke(config)
    stop1 = time.time()

    start2 = time.time()
    json_output_dir = i2b2_conll2jsonlines.invoke(conll_output_dir)
    stop2 = time.time()

    logger.info("*" * 60)
    logger.info(f"CoNLL format dir: {conll_output_dir}")
    logger.info(f"Time: {stop1-start1:.2f}s")
    logger.info(f"JSON format dir: {json_output_dir}")
    logger.info(f"Time: {stop2-start2:.2f}s")
    logger.info(f"Total time: {stop2-start1}s")
    logger.info("*" * 60)


if __name__ == "__main__":
    main()
