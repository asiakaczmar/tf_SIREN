import tensorflow as tf
import itertools
import numpy as np
import os

#For now we will assume width and height are the same
ROWS_COLS = 32
BATCH_SIZE = 512
EPOCHS = 600
LATENT_DIM = 256
NOISE_LOC = 0
NOISE_STD = 0.2
PIXEL_NUMBER = ROWS_COLS*ROWS_COLS


@tf.function
def add_noise(image):
    noise = tf.random.normal(shape=tf.shape(image), mean=NOISE_LOC, stddev=NOISE_STD, dtype=tf.float32   )
    noisy_img = image + noise
    noisy_img -= tf.math.reduce_min(noisy_img)
    return noisy_img/(tf.math.reduce_max(noisy_img))

@tf.function
def build_train_tensors(ds):
    img_mask_idx = np.array(list(itertools.product(range(ROWS_COLS), range(ROWS_COLS))))
    original_image = tf.cast(ds['image'], tf.float32) / 255.
    noisy_image = add_noise(original_image)
    noisy_image = tf.gather_nd(noisy_image, img_mask_idx)
    original = tf.gather_nd(original_image, img_mask_idx)
    img_mask = tf.cast(img_mask_idx, tf.float32) / ROWS_COLS
    return original_image, img_mask, noisy_image, original

def process_ds(ds, shuffle=True, dataset_len=None, batch_size=BATCH_SIZE):
    ds = ds.map(build_train_tensors, num_parallel_calls=2 * os.cpu_count())
    if shuffle:
        ds = ds.shuffle(dataset_len)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds