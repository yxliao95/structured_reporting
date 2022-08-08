import logging, os, hydra, timeit

logger = logging.getLogger()


@hydra.main(version_base=None, config_path="../config", config_name="data_preprocessing")
def main(config):
    print(config)


if __name__ == "__main__":
    main()
