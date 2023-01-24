from omegaconf import OmegaConf
from jax import random

# # ref: https://python-patterns.guide/gang-of-four/singleton/
# class _Config(object):
#     _instance = None

#     def __new__(cls,params={}):
#         if cls._instance is None:
#             print('Creating Config!!')
#             cls._instance = super(_Config, cls).__new__(cls)

#             conf_file = OmegaConf.load('configs/default.yaml')
#             conf_local = OmegaConf.create(params)
#             conf_cli = OmegaConf.from_cli()
#             cls._instance.conf = OmegaConf.merge(conf_file, conf_local, conf_cli)
#         return cls._instance

class _Config():
    def __init__(self, params):
        conf_file = OmegaConf.load('configs/default.yaml')
        conf_local = OmegaConf.create(params)
        conf_cli = OmegaConf.from_cli()
        self.conf = OmegaConf.merge(conf_file, conf_local, conf_cli)


# Config({"mode": "train", "seed": 2}).conf
# Config().conf.seed = 3

# ref:  https://stackoverflow.com/a/5517322
class Env():
    def __init__(self, params={}):
        self.conf = _Config(params).conf

    def __getattr__(self, key):
        try:
            return self.conf[key]
        except:
            return self.conf.base[key]

env = Env()

