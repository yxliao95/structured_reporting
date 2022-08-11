import logging, os, hydra, time
import mimic_cxr_raw2conll


logger = logging.getLogger()
pkg_path = os.path.dirname(__file__)
coref_module_path = os.path.dirname(pkg_path)
config_path = os.path.join(os.path.dirname(coref_module_path), "config")


@hydra.main(version_base=None, config_path=config_path, config_name="coreference_resolution")
def main(config):

    # print(OmegaConf.to_yaml(config))

    src = os.path.join(config.fast_coref_dir, "src")
    if not os.path.exists(src):
        logger.error(
            f"Directory 'fast-coref/' not found. Please clone the 'fast-coref' repo to {coref_module_path}, e.g.: git clone git@github.com:liaoooyx/fast-coref.git"
        )
        raise Exception(f"Lacking the fast-coref repo in {coref_module_path}")

    config = config.coref_data_preprocessing.mimic_cxr
    start1 = time.time()
    conll_output_dir = mimic_cxr_raw2conll.invoke(config)
    stop1 = time.time()

    start2 = time.time()
    # json_output_dir = i2b2_conll2jsonlines.invoke(conll_output_dir)
    stop2 = time.time()

    # logger.info("*" * 60)
    # logger.info(f"CoNLL format dir: {conll_output_dir}")
    # logger.info(f"Time: {stop1-start1:.2f}s")
    # logger.info(f"JSON format dir: {json_output_dir}")
    # logger.info(f"Time: {stop2-start2:.2f}s")
    # logger.info(f"Total time: {stop2-start1}s")
    # logger.info("*" * 60)


if __name__ == "__main__":
    main()
