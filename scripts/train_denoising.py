import itertools
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds

from tf_siren.hypernet import NeuralProcessHyperNet

#For now we will assume width and height are the same
ROWS_COLS = 32
BATCH_SIZE = 512
EPOCHS = 600
LATENT_DIM = 256
NOISE_LOC = 0
NOISE_STD = 0.2
PIXEL_NUMBER = ROWS_COLS*ROWS_COLS

ds, ds_info = tfds.load('cifar10', data_dir='/cluster/work/scopem/kjoanna/tensorflow_datasets',  split='train[:-10%]', with_info=True)  # type: tf.data.Dataset
val_ds, _ = tfds.load('cifar10', data_dir='/cluster/work/scopem/kjoanna/tensorflow_datasets', split='train[-10%:]', with_info=True)

input_shape = ds_info.features['image'].shape
train_dataset_len = int(ds_info.splits['train'].num_examples * 0.9)

rows, cols, channels = input_shape
pixel_count = rows * cols


@tf.function
def add_noise(image):
    noise = tf.random.normal(shape=tf.shape(image), mean=NOISE_LOC, stddev=NOISE_STD, dtype=tf.float32   )
    noisy_img = image + noise
    noisy_img -= tf.math.reduce_min(noisy_img)
    return noisy_img/(tf.math.reduce_max(noisy_img))

@tf.function
def build_train_tensors(ds):
    img_mask_idx = np.array(list(itertools.product(range(ROWS_COLS), range(ROWS_COLS))))
    original = tf.cast(ds['image'], tf.float32) / 255.
    noisy_image = add_noise(original)
    noisy_image = tf.gather_nd(noisy_image, img_mask_idx)
    original = tf.gather_nd(original, img_mask_idx)
    img_mask = tf.cast(img_mask_idx, tf.float32) / ROWS_COLS
    return img_mask, noisy_image, original


def process_ds(ds, shuffle=True, dataset_len=None):
    ds = ds.map(build_train_tensors, num_parallel_calls=2 * os.cpu_count())
    if shuffle:
        ds = ds.shuffle(dataset_len)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

ds = process_ds(ds, dataset_len=train_dataset_len)
val_ds = process_ds(val_ds, shuffle=False)

# Build model
model = NeuralProcessHyperNet(
    input_units=2, output_units=channels,  # number of inputs and outputs
    siren_units=LATENT_DIM, hyper_units=LATENT_DIM, latent_dim=LATENT_DIM,  # number of units
    num_siren_layers=3, num_hyper_layers=1, num_encoder_layers=2,  # number of layers
    encoder_activation='sine', hyper_activation='relu', final_activation='sigmoid',  # activations
    lambda_embedding=0.1, lambda_hyper=100., lambda_mse=100.0,  # Loss scaling
)
# instantiate model
dummy_input = [tf.zeros([BATCH_SIZE, PIXEL_NUMBER, 2]), tf.zeros([BATCH_SIZE, PIXEL_NUMBER, 3]), tf.zeros([BATCH_SIZE, PIXEL_NUMBER, 3])]
_ = model(dummy_input)

model.summary()

BATCH_SIZE = min(BATCH_SIZE, ROWS_COLS*ROWS_COLS)
num_steps = int(train_dataset_len * EPOCHS / BATCH_SIZE)
print("Total training steps : ", num_steps)
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.00005, decay_steps=num_steps, end_learning_rate=0.00002,
                                                              power=2.0)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)


@tf.function
def loss_func(y_true, y_pred):
    # Note: This loss is slightly different from the paper.
    # Note: This loss is MSE * channels. To compute true MSE, divide the loss value by number of channels.
    diff = 1.0 / (PIXEL_NUMBER) * (tf.reduce_sum(tf.square(y_true - y_pred), axis=[1, 2]))
    diff = tf.reduce_mean(diff)
    return diff


model.compile(optimizer, loss=loss_func, run_eagerly=False)

checkpoint_dir = 'checkpoints/cifar10/denoising/'
checkpoint_path = checkpoint_dir + 'model'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


if os.path.exists(checkpoint_dir + 'checkpoint'):
    print("Loaded weights for continued training !")
    model.load_weights(checkpoint_path)

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logdir = os.path.join('../logs/cifar10/denoising/', timestamp)

if not os.path.exists(logdir):
    os.makedirs(logdir)

callbacks = [
    # Select lowest pixel mse loss as checkpoint saver.
    tf.keras.callbacks.ModelCheckpoint(checkpoint_dir + 'model', monitor='image_loss', verbose=0,
                                       save_best_only=True, save_weights_only=True, mode='min'),
    tf.keras.callbacks.TensorBoard(logdir, update_freq='batch', profile_batch='500,520'),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=2, verbose=1)
]

model.fit(ds, epochs=EPOCHS, callbacks=callbacks, verbose=1, validation_data=val_ds)
