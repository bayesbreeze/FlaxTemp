# python main.py mode=train
# exec(open("main.py").read())

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import utils
import run_lib as run
from tqdm import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from configs.Config import env


def main():
    run.train()

if __name__ == '__main__':
    main()


