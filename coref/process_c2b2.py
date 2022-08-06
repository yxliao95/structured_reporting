import c2b2_raw2conll as c2b2_raw2conll
import c2b2_conll2jsonlines as c2b2_conll2jsonlines
import logging, os, hydra, timeit

logger = logging.getLogger()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    current = os.path.dirname(os.path.realpath(__file__))
    src = os.path.join(current, "fast-coref", "src")
    if not os.path.exists(src):
        logger.error(
            f"Directory ./coref/fast-coref not found. Please clone the 'fast-coref' repo to {current}, e.g.: git clone git@github.com:liaoooyx/fast-coref.git"
        )
        raise Exception(f"Lacking the fast-coref repo in {current}")

    start1 = timeit.default_timer()
    conll_output_dir = c2b2_raw2conll.invoke(config)
    stop1 = timeit.default_timer()

    start2 = timeit.default_timer()
    json_output_dir = c2b2_conll2jsonlines.invoke(conll_output_dir)
    stop2 = timeit.default_timer()

    logger.info("*" * 30)
    logger.info(f"CoNLL format dir: {conll_output_dir}")
    logger.info(f"Time: {stop1-start1:.2f}s")
    logger.info(f"JSON format dir: {json_output_dir}")
    logger.info(f"Time: {stop2-start2:.2f}s")
    logger.info(f"Total time: {stop2-start1}s")
    logger.info("*" * 30)


if __name__ == "__main__":
    main()
