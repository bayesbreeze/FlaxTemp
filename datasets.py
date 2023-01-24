# exec(open('common/dataset.py').read())
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
from configs.Config import env


tf.config.set_visible_devices([], "GPU")

def showInfo(ds, info):
    # tfds.as_dataframe(ds.take(4), info)
    ds = ds.shuffle(100)
    fig = tfds.show_examples(ds, info)
    print("===============")
    print("features: ", info.features)
    print("splits: ", list(info.splits.keys()))
    print("lables#: ", info.features["label"].num_classes , info.features["label"].names)
    print("splits#: ", info.splits['train'].num_examples) #'train[15%:75%]'

def testPerformance(ds):
    ds = ds.map(prepare_image).cache()
    ds = ds.batch(32, drop_remainder=True).prefetch(-1)
    tfds.benchmark(ds, batch_size=32)
    tfds.benchmark(ds, batch_size=32) # faster!

def prepare_image(x):
    x = tf.cast(x['image'], tf.float32)
    x = tf.reshape(x, (-1,))
    return x

# ref：https://www.tensorflow.org/datasets/overview#load_a_dataset
# ref：https://github.com/google/flax/blob/main/examples/vae/train.py
# usage 1
# for image in train_ds:
#     print(type(image), image.shape)

# usage 2
# steps_per_epoch = conf.data.train_num // conf.train.batch_size
# for epoch in range(conf.train.num_epochs):
#     print(epoch)
#     for _ in range(steps_per_epoch):
#         batch = next(train_ds)
#         # print(type(batch), batch.shape)
def get_dataset():
    # tfds.list_builders() # list all dataset available
 
    train_ds, info = tfds.load(
            env.data.dataset, 
            split=tfds.Split.TRAIN, 
            shuffle_files=True, 
            as_supervised=False, 
            with_info=True)
    # showInfo(train_ds, info)
    # testPerformance(train_ds)

    env.train.num_steps_per_epoch = info.splits[tfds.Split.TRAIN].num_examples // env.train.batch_size

    train_ds = train_ds.map(prepare_image).cache() 
    # train_ds = train_ds.repeat(env.train.num_epochs).shuffle(env.data.shuffle_buff)
    train_ds = train_ds.repeat().shuffle(env.data.shuffle_buff)
    # train_ds = train_ds.shuffle(env.data.shuffle_buff)
    train_ds = train_ds.batch(env.train.batch_size, drop_remainder=True).prefetch(-1)
    train_ds = iter(tfds.as_numpy(train_ds))

    test_ds = tfds.load(env.data.dataset, split=tfds.Split.TEST)
    test_ds = test_ds.map(prepare_image).batch(info.splits[tfds.Split.TEST].num_examples)
    test_ds = np.array(list(test_ds)[0])
    test_ds = jax.device_put(test_ds)

    return train_ds, test_ds

