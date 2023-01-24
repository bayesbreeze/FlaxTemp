import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax
from clu import metrics
import tensorflow as tf
import tensorflow_datasets as tfds

import utils
import models

import run_lib as run
from tqdm import tqdm
from configs.Config import env
from flax.training import checkpoints

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


def compute_metrics(recon_x, x, mean, logvar):
    bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean() # reconstruction
    kld_loss = kl_divergence(mean, logvar).mean()  # prior error
    return {
        'bce': bce_loss,
        'kld': kld_loss,
        'loss': bce_loss + kld_loss
    }

env.model = models.VAE(latents=env.train.latents)

@jax.jit
def train_step(state, batch, z_rng):
    def loss_fn(params):
        recon_x, mean, logvar = env.model.apply({'params': params}, batch, z_rng)

        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + kld_loss
        return loss, 1
    (loss, val), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return loss, state.apply_gradients(grads=grads)


@jax.jit
def eval(params, images, z, z_rng):
    def eval_model(vae):
        recon_images, mean, logvar = vae(images, z_rng)
        comparison = jnp.concatenate([images[:8].reshape(-1, 28, 28, 1),
                                      recon_images[:8].reshape(-1, 28, 28, 1)])

        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, 28, 28, 1)
        metrics = compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images

    return nn.apply(eval_model, env.model)({'params': params})


def train():
    # 1. init state
    h = utils.TrainHelper()
    rng = random.PRNGKey(0)
    rng, key = random.split(rng)
    init_data = jnp.ones((env.train.batch_size, 784), jnp.float32)
    state = train_state.TrainState.create(
        apply_fn=env.model.apply,
        params=env.model.init(key, init_data, rng)['params'],
        tx=optax.adam(env.train.learning_rate),
    )
    state = h.restore_checkpoint(state)

    # 2. pre-run
    rng, z_key, eval_rng = random.split(rng, 3)
    z = random.normal(z_key, (64, env.train.latents))
    
    # 3. run
    for step in range(h.start_step, env.train.n_iters):
        batch = next(h.train_ds)
        rng, key = random.split(rng)
        loss, state = train_step(state, batch, key)
        
        if jax.process_index() != 0:
            continue
        
        # 4. deal with output
        h.update(step)
        if step % env.train.log_freq == 0:
            h.log_train_loss(loss)
            
        if step % env.train.eval_freq == 0:
            metrics, comparison, sample = eval(state.params, h.test_ds, z, eval_rng)
            h.log_eval_loss(metrics['loss'])
        
        if step > 0 and step % env.train.snapshot_freq == 0 or step == env.train.n_iters-1:
            metrics, comparison, sample = eval(state.params, h.test_ds, z, eval_rng)
            h.save_snapshot(step, state, metrics, comparison, sample)

    h.close()






