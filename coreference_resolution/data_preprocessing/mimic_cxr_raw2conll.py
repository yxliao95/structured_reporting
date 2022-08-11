from omegaconf import OmegaConf


def invoke(config):
    print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    invoke()
