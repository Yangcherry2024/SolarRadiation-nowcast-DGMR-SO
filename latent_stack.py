import functools
import layers
import tensorflow.compat.v1 as tf
import sonnet as snt


class LatentCondStack(snt.Module):
    """Latent Conditioning Stack for the Sampler."""

    def __init__(self):
        super().__init__(name=None)
        self._conv1 = layers.SNConv2D(output_channels=8, kernel_size=3)
        self._lblock1 = LBlock(output_channels=24, input_channels=8)
        self._lblock2 = LBlock(output_channels=48, input_channels=24)
        self._lblock3 = LBlock(output_channels=192, input_channels=48)
        self._mini_attn_block = Attention(num_channels=192)
        #self._lblock4 = LBlock(output_channels=768, input_channels=192)
        self._lblock4 = LBlock(output_channels=768, input_channels=192)

    def __call__(self, batch_size, resolution=(126, 208), is_training=True):

        # Independent draws from a Normal distribution.
        h, w = resolution[0] // 32, resolution[1] // 32
        z = tf.random.normal([batch_size, h, w, 8])

        # 3x3 convolution.
        z = self._conv1(z, is_training=is_training)

        # Three L Blocks to increase the number of channels to 24, 48, 192.
        z = self._lblock1(z)
        z = self._lblock2(z)
        z = self._lblock3(z)

        # Spatial attention module.
        z = self._mini_attn_block(z)

        # L Block to increase the number of channels to 768.
        z = self._lblock4(z)

        return z


class LBlock(snt.Module):
    """Residual block for the Latent Stack."""

    def __init__(self, output_channels, input_channels, kernel_size=3, conv=layers.Conv2D,
                 activation=tf.nn.relu):
        """Constructor for the D blocks of the DVD-GAN.

        Args:
          output_channels: Integer number of channels in convolution operations in
            the main branch, and number of channels in the output of the block.
          kernel_size: Integer kernel size of the convolutions. Default: 3.
          conv: TF module. Default: layers.Conv2D.
          activation: Activation before the conv. layers. Default: tf.nn.relu.
        """
        super().__init__(name=None)
        self._output_channels = output_channels
        self._input_channels = input_channels
        self._kernel_size = kernel_size
        self._conv1 = conv(output_channels=self._output_channels,
                           kernel_size=self._kernel_size)
        self._conv2 = conv(output_channels=self._output_channels,
                           kernel_size=self._kernel_size)
        self._conv_residual = conv(output_channels=self._output_channels - self._input_channels,
                                   kernel_size=1)
        self._activation = activation

    def __call__(self, inputs):
        """Build the LBlock.

        Args:
          inputs: a tensor with a complete observation [N 256 256 1]

        Returns:
          A tensor with discriminator loss scalars [B].
        """

        # Stack of two conv. layers and nonlinearities that increase the number of
        # channels.
        h0 = self._activation(inputs)
        h1 = self._conv1(h0)
        h1 = self._activation(h1)
        h2 = self._conv2(h1)

        # Prepare the residual connection branch.
        input_channels = h0.shape.as_list()[-1]
        if input_channels < self._output_channels:
            sc = self._conv_residual(inputs)
            sc = tf.concat([inputs, sc], axis=-1)
        else:
            sc = inputs

        # Residual connection.
        return h2 + sc


def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    k = tf.reshape(k, [-1, k.shape[-1]])  # [h, w, c] -> [L, c]
    v = tf.reshape(v, [-1, v.shape[-1]])  # [h, w, c] -> [L, c]

    # Einstein summation corresponding to the query * key operation.
    beta = tf.nn.softmax(tf.einsum('hwc, Lc->hwL', q, k), axis=-1)

    # Einstein summation corresponding to the attention * value operation.
    out = tf.einsum('hwL, Lc->hwc', beta, v)
    return out


class Attention(snt.Module):
    """Attention module."""

    def __init__(self, num_channels, ratio_kq=8, ratio_v=8, conv=layers.Conv2D):
        """Constructor."""
        super().__init__(name=None)
        self._num_channels = num_channels
        self._ratio_kq = ratio_kq
        self._ratio_v = ratio_v
        self._conv1 = conv(
            output_channels=self._num_channels // self._ratio_kq,
            kernel_size=1, padding='VALID', use_bias=False)
        self._conv2 = conv(
            output_channels=self._num_channels // self._ratio_kq,
            kernel_size=1, padding='VALID', use_bias=False)
        self._conv3 = conv(
            output_channels=self._num_channels // self._ratio_v,
            kernel_size=1, padding='VALID', use_bias=False)
        self._conv4 = conv(
            output_channels=self._num_channels,
            kernel_size=1, padding='VALID', use_bias=False)

        # Learnable gain parameter
        self._gamma = tf.get_variable(
            'miniattn_gamma', shape=[1], trainable=True, dtype=tf.float32)

    def __call__(self, tensor):
        # Compute query, key and value using 1x1 convolutions.
        query = self._conv1(tensor)
        key = self._conv2(tensor)
        value = self._conv3(tensor)

        # Apply the attention operation.
        out = layers.ApplyAlongAxis2(functools.partial(
            attention_einsum, k=key, v=value), axis=0)(query)
        out = self._gamma * self._conv4(out)

        # Residual connection.
        return out + tensor
