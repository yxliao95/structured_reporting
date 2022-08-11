import logging
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger()

def add_config_item(config: DictConfig, key: str, value: object) -> DictConfig:
    """ Convert ``DictConfig`` to ``dict``, add key and value, convert it back to ``DictConfig``. 
    See https://omegaconf.readthedocs.io/en/2.0_branch/usage.html for more information.
    """
    config = OmegaConf.to_container(config)
    config[key] = value
    return OmegaConf.create(config)

def logDebug_config(config: DictConfig):
    logger.debug(OmegaConf.to_yaml(config))

def print_config(config: DictConfig):
    print(OmegaConf.to_yaml(config))