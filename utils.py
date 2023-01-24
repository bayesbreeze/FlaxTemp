import sys
import os
import io
import math
from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
import flax
import flax.jax_utils as flax_utils
import numpy as np
from PIL import Image
import tensorflow as tf
from flax.metrics import tensorboard
import datetime
import matplotlib.pyplot as plt
from configs.Config import env
from flax.training import checkpoints
import datasets
import logging

# refer: https://github.com/google/flax/blob/main/examples/vae/utils.py
def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
    """Make a grid of images and Save it into an image file.
  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp - A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    scale_each (bool, optional): If ``True``, scale each image in the batch of
      images separately rather than the (min, max) over all images. Default: ``False``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the filename extension.
      If a file object was used instead of a filename, this parameter should always be used.
  """
    if not (isinstance(ndarray, jnp.ndarray) or
            (isinstance(ndarray, list) and all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError(f'array_like of tensors expected, got {type(ndarray)}')

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] +
                        padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full((height * ymaps + padding, width * xmaps +
                    padding, num_channels), pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[y * height + padding:(y + 1) * height,
                           x * width + padding:(x + 1) * width].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format)

# refer: https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(images, labels=None):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        # plt.subplot(5, 5, i + 1, title=class_names[train_labels[i]])
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

    return figure


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)

# https://github.com/tqdm/tqdm/issues/724
class DummyTqdmFile(object):
    """ Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


class TrainHelper:
    conf: Any
    pbar: Any
    step: int
    num_steps_total: int
    num_steps_per_epoch: int
    train_ds: Any
    test_ds: Any
    state: Any

    def __init__(self):
        self.conf = env.conf

        #  prepare dataset
        self.train_ds, self.test_ds = datasets.get_dataset()
        self.num_steps_per_epoch = env.train.num_steps_per_epoch
        #  = self.num_steps_total // env.train.num_epochs
        self.pbar = tqdm(total=env.train.n_iters, leave=True, desc="train initilizing...")

        logging.basicConfig(level=logging.DEBUG, stream=DummyTqdmFile(sys.stderr))
        self.log = logging.getLogger("train")
        
        # rng = jax.random.PRNGKey(env.seed)

        # parepare dirs
        Path(env.sample_dir).mkdir(parents=True, exist_ok=True)
        Path(env.tb_dir).mkdir(parents=True, exist_ok=True)
        Path(env.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base_log_dir = "%s/logs/%s" % (env.base.tb_dir, current_time)
        train_log_dir = base_log_dir + '/train'
        # test_log_dir = base_log_dir + '/test'
        self.summary_writer = tensorboard.SummaryWriter(train_log_dir)
        

        self.train_loss = -1
        self.eval_loss = -1
        
    def restore_checkpoint(self, state):
        # prepare state
        self.state = checkpoints.restore_checkpoint(env.checkpoint_dir, state)
        self.start_step = int(self.state.step)
        self.pbar.update(self.start_step)
        return self.state

    # def __getattr__(self, key):
    #     try:
    #         return self.conf.train[key]
    #     except:
    #         raise Exception("Key '%s' is not found in conf.train" % key)

    def update(self, step):
        self.step = step
        self.pbar.update(1)
        self.pbar.set_description("loss: %.2f, eval: %.2f"%(self.train_loss,self.eval_loss))

    def log_train_loss(self, loss):
        self.train_loss = loss
        self.summary_writer.scalar('loss', loss, self.step)

    def log_eval_loss(self, loss):
        self.eval_loss = loss
        self.summary_writer.scalar('eval_loss', loss, self.step)
    
    def save_snapshot(self, step, state, metrics, comparison, sample):
        epoch = (self.step+1) // self.num_steps_per_epoch
         # state = state.replace(rng=rng)
        checkpoints.save_checkpoint(env.checkpoint_dir, state,
                    step=self.step,
                    keep=env.train.snapshot_keep)

        if env.train.snapshot_sampling:
            figure = image_grid(np.reshape(sample[0:25], (-1, 28, 28, 1)))
            self.summary_writer.image("sample", plot_to_image(figure), step=step)

            save_image(comparison, 
                    "%s/reconstruction_%d.png"%(env.sample_dir, epoch), nrow=8)
            save_image(sample, 
                    "%s/sample_%d.png"%(env.sample_dir, epoch), nrow=8)

    def close(self):
        self.summary_writer.flush()
        self.summary_writer = None
        self.pbar.close()
        self.state = None
        self.train_ds = None
        self.test_ds = None
