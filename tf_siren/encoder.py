import tensorflow as tf
from tf_siren import siren
from tensorflow.keras.layers import Conv2D, Input, Dense, Reshape, Conv2DTranspose,\
   Activation, BatchNormalization, ReLU, Concatenate, Flatten
from tensorflow.keras.models import Model

class ConvEncoder(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        self.latent_dim = 256
        super().__init__(*args, **kwargs)
        self.inp = Input(shape=(32, 32, 3), name='dae_input')
        conv_block1 = self.conv_block(self.inp, 32, 3)
        conv_block2 = self.conv_block(conv_block1, 64, 3)
        conv_block3 = self.conv_block(conv_block2, 128, 3)
        conv_block4 = self.conv_block(conv_block3, 256, 3)
        conv_block5 = self.conv_block(conv_block4, 256, 3)
        self.out = Flatten()(conv_block5)
        self.model = Model(self.inp, self.out, name='dae')

    @staticmethod
    def conv_block(x, filters, kernel_size, strides=2):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    @tf.function
    def call(self, input, training=None, mask=None):
        return self.model(input)

class SetEncoder(tf.keras.Model):

    def __init__(self, output_units: int, hidden_units: int, num_hidden_layers: int = 1,
                 activation: str = 'sine', w0: float = 30.0, use_bias: bool = True,
                 name='SetEncoder', **kwargs):
        """
        SetEncoder from the paper [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661).

        Args:
            output_units: Positive integer, dimensionality of the output space.
            hidden_units: Positive integer, dimensionality of the hidden space.
            num_hidden_layers: Number of layers in the Set Encoder network.
            activation: Activation function of the Set Encoder layers.
            w0: w0 in the activation step `act(x; w0) = sin(w0 * x)` iff "sine"
                activation function is used. Ignored otherwise.
            use_bias: Boolean whether to use bias or not.

        # References:
            -   [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
        """
        super().__init__(name=name, **kwargs)

        assert activation in ['relu', 'sine'], "`activation` must be either relu or sine"

        self.latent_dim = output_units

        if activation == 'relu':
            activation = tf.keras.layers.Activation(activation)
            kernel_initializer = 'he_normal'
        else:
            activation = siren.Sine(w0)
            kernel_initializer = 'siren_uniform'

        bias_initializer = 'siren_first_uniform'

        net = [siren.SinusodialRepresentationDense(hidden_units, activation=activation, w0=w0,
                                                   kernel_initializer=kernel_initializer,
                                                   bias_initializer=bias_initializer,
                                                   use_bias=use_bias, name=name, **kwargs)]

        for _ in range(num_hidden_layers):
            net.append(siren.SinusodialRepresentationDense(hidden_units, activation=activation, w0=w0,
                                                           kernel_initializer=kernel_initializer,
                                                           bias_initializer=bias_initializer,
                                                           use_bias=use_bias, name=name, **kwargs))

        net.append(siren.SinusodialRepresentationDense(output_units, activation=activation, w0=w0,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       use_bias=use_bias, name=name, **kwargs))

        self.net = tf.keras.Sequential(net)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        coords, pixels, _ = inputs
        inputs = tf.concat([coords, pixels], axis=-1)
        embedding = self.net(inputs)

        # [B, num_sampled_pixels, embedding_dim] -> [B, embedding_dim]
        embedding = tf.reduce_mean(embedding, axis=1)

        return embedding

if __name__ == '__main__':
    import numpy as np
    encoder = ConvEncoder()
    out = encoder(np.random.uniform(0,1,[1, 32, 32, 3]))
    print(out.shape)