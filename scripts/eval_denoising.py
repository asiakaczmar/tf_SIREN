import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_siren.hypernet import NeuralProcessHyperNet
import tensorflow_datasets as tfds


from scripts.utils import process_ds, LATENT_DIM, PIXEL_NUMBER

test_ds, ds_info = tfds.load('cifar10', data_dir='/cluster/work/scopem/kjoanna/tensorflow_datasets', split='test', with_info=True)
original_test_ds = test_ds
input_shape = ds_info.features['image'].shape
rows, cols, channels = input_shape
pixel_count = rows * cols
test_dataset_len = ds_info.splits['test'].num_examples



test_ds = process_ds(test_ds, dataset_len=test_dataset_len, batch_size=1)


# Build model
model = NeuralProcessHyperNet(
    input_units=2, output_units=channels,  # number of inputs and outputs
    siren_units=LATENT_DIM, hyper_units=LATENT_DIM, latent_dim=LATENT_DIM,  # number of units
    num_siren_layers=3, num_hyper_layers=1, num_encoder_layers=2,  # number of layers
    encoder_activation='sine', hyper_activation='relu', final_activation='sigmoid',  # activations
    lambda_embedding=0.1, lambda_hyper=100., lambda_mse=100.0, encoder='conv' # Loss scaling
)


# Restore model
checkpoint_dir = 'checkpoints/cifar10/denoising/'
checkpoint_path = checkpoint_dir + 'model'
if len(glob.glob(checkpoint_dir + "*.index")) == 0:
    raise FileNotFoundError("Model checkpoint not found !")

# instantiate model
dummy_input = [tf.zeros([1, PIXEL_NUMBER, 2]), tf.zeros([1, PIXEL_NUMBER, 3]), tf.zeros([1, PIXEL_NUMBER, 3])]

#_ = model([tf.zeros([1, 1, 2]), tf.zeros([1, 1, 1]), tf.zeros([1, 1, 1])])

# load checkpoint
model.load_weights(checkpoint_path).expect_partial()  # skip optimizer loading


for one_set in test_ds.take(5):
    
    
    predicted_image, _ = model(one_set)
    predicted_image = np.array(predicted_image)
    predicted_image = predicted_image.reshape((rows, cols, channels))  # type: np.ndarray
    #predicted_image = predicted_image.clip(0.0, 1.0)
    
    
    
    fig, axes = plt.subplots(1, 2)
    plt.sca(axes[0])
    plt.imshow(one_set[0].numpy().reshape(rows, cols, channels))
    plt.title("Ground Truth Image")

    
    plt.sca(axes[1])
    plt.imshow(predicted_image)
    plt.title("Predicted Image")
    
    fig.tight_layout()
    plt.show()
