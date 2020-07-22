import itertools
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds

from tf_siren.hypernet import NeuralProcessHyperNet

from scripts.utils import process_ds, LATENT_DIM, BATCH_SIZE, ROWS_COLS, PIXEL_NUMBER, EPOCHS

ds, ds_info = tfds.load('cifar10', data_dir='/cluster/work/scopem/kjoanna/tensorflow_datasets', split='train[:-10%]', with_info=True)  # type: tf.data.Dataset
val_ds, _ = tfds.load('cifar10', data_dir='/cluster/work/scopem/kjoanna/tensorflow_datasets', split='train[-10%:]', with_info=True)

input_shape = ds_info.features['image'].shape
train_dataset_len = int(ds_info.splits['train'].num_examples * 0.9)

rows, cols, channels = input_shape
pixel_count = rows * cols



ds = process_ds(ds, dataset_len=train_dataset_len)
val_ds = process_ds(val_ds, shuffle=False)

# Build model
model = NeuralProcessHyperNet(
    input_units=2, output_units=channels,  # number of inputs and outputs
    siren_units=LATENT_DIM, hyper_units=LATENT_DIM, latent_dim=LATENT_DIM,  # number of units
    num_siren_layers=3, num_hyper_layers=1, num_encoder_layers=2,  # number of layers
    encoder_activation='sine', hyper_activation='relu', final_activation='sigmoid',  # activations
    lambda_embedding=0.1, lambda_hyper=100., lambda_mse=100., # Loss scaling
    encoder='conv'
)
# instantiate model
dummy_input = [tf.zeros([BATCH_SIZE, ROWS_COLS, ROWS_COLS, 3]), tf.zeros([BATCH_SIZE, PIXEL_NUMBER, 2]), tf.zeros([BATCH_SIZE, PIXEL_NUMBER, 3]), tf.zeros([BATCH_SIZE, PIXEL_NUMBER, 3])]
_ = model(dummy_input)

model.summary()

BATCH_SIZE = min(BATCH_SIZE, ROWS_COLS*ROWS_COLS)
num_steps = int(train_dataset_len * EPOCHS / BATCH_SIZE)
print("Total training steps : ", num_steps)
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.00005, decay_steps=num_steps, end_learning_rate=0.00002,
                                                              power=2.0)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

@tf.function
def loss_mae(y_true, y_pred):
    # Note: This loss is slightly different from the paper.
    # Note: This loss is MSE * channels. To compute true MSE, divide the loss value by number of channels.
    diff = 1.0 / (PIXEL_NUMBER) * (tf.reduce_sum(tf.math.abs(y_true - y_pred), axis=[1, 2]))
    diff = tf.reduce_mean(diff)
    return diff

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


#if os.path.exists(checkpoint_dir + 'checkpoint'):
#    print("Loaded weights for continued training !")
#    model.load_weights(checkpoint_path)

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

