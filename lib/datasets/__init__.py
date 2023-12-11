from lib.utils.builder import build_dataset
from yacs.config import CfgNode as CN

from .grabnet_data import GrabNetData
from .oishape import OIShape


def create_dataset(cfg: CN, data_preset: CN, **kwargs):
    """
    Create a dataset instance.
    """
    if cfg.TYPE == "MixDataset":
        # list of CN of each dataset
        if isinstance(cfg.CONTENT, dict):
            dataset_list = [v for k, v in cfg.CONTENT.items()]
        else:
            dataset_list = cfg.CONTENT
        return MixDataset(dataset_list, data_preset, cfg.RATIO)
    else:
        # default building from cfg
        return build_dataset(cfg, data_preset=data_preset, **kwargs)
