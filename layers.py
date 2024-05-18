import tensorflow as tf
import sonnet as snt
import functools
from keras import layers
from sonnet.src.conv import ConvND


def downsample_avg_pool(x):
    """Utility function for downsampling by 2x2 average pooling."""
    return layers.AveragePooling2D(2, 2, data_format='channels_last')(x)


def downsample_avg_pool3d(x):
    """Utility function for downsampling by 2x2 average pooling."""
    return layers.AveragePooling3D(2, 2, data_format='channels_last')(x)


def upsample_nearest_neighbor(inputs, upsample_size):
    """Nearest neighbor upsampling.

    Args:
      inputs: inputs of size [b, h, w, c] where b is the batch size, h the height,
        w the width, and c the number of channels.
      upsample_size: upsample size S.
    Returns:
      outputs: nearest neighbor upsampled inputs of size [b, s * h, s * w, c].
    """
    return tf.image.resize(inputs, [upsample_size*inputs.shape[1], upsample_size*inputs.shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


class Conv2D(snt.Module):
    """2D convolution."""

    def __init__(self, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', use_bias=True):
        """Constructor."""

        super().__init__(name=None)
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._rate = rate
        self._padding = padding
        self._initializer = snt.initializers.Orthogonal()
        self._use_bias = use_bias
        self._conv2D = snt.Conv2D(
            output_channels=self._output_channels,
            kernel_shape=self._kernel_size,
            stride=self._stride,
            rate=self._rate,
            padding=self._padding,
            with_bias=self._use_bias,
            w_init=self._initializer,
            b_init=None
        )

    def __call__(self, tensor):
        # TO BE IMPLEMENTED
        # One possible implementation is provided in the Sonnet library: snt.Conv2D.
        return self._conv2D(tensor)


class SNConv2D(ConvND):
    """2D convolution with spectral normalisation."""

    def __init__(self, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', sn_eps=0.0001, use_bias=True):
        """Constructor."""
        super().__init__(
            num_spatial_dims=2,
            output_channels=output_channels,
            kernel_shape=kernel_size,
            stride=stride,
            rate=rate,
            padding=padding,
            with_bias=use_bias,
            w_init=snt.initializers.Orthogonal(),
            b_init=None,
            data_format="NHWC",
            name=None)
        self._spectral_normalizer = SpectralNormalizer(epsilon=sn_eps)

    def __call__(self, tensor, is_training=True):

        self._initialize(tensor)

        if self.padding_func:
            tensor = tf.pad(tensor, self._padding)

        normed_w = self._spectral_normalizer(self.w, is_training=is_training)

        outputs = tf.nn.convolution(
            tensor,
            normed_w,
            strides=self.stride,
            padding=self.conv_padding,
            dilations=self.rate,
            data_format=self.data_format)

        if self.with_bias:
            outputs = tf.nn.bias_add(
                outputs, self.b, data_format=self.data_format)

        return outputs


class SNConv3D(ConvND):
    """2D convolution with spectral normalisation."""

    def __init__(self, output_channels, kernel_size, stride=1, rate=1,
                 padding='SAME', sn_eps=0.0001, use_bias=True):
        """Constructor."""
        super().__init__(
            num_spatial_dims=3,
            output_channels=output_channels,
            kernel_shape=kernel_size,
            stride=stride,
            rate=rate,
            padding=padding,
            with_bias=use_bias,
            w_init=snt.initializers.Orthogonal(),
            b_init=None,
            data_format="NDHWC",
            name=None)
        self._spectral_normalizer = SpectralNormalizer(epsilon=sn_eps)

    def __call__(self, tensor, is_training=True):

        self._initialize(tensor)

        if self.padding_func:
            tensor = tf.pad(tensor, self._padding)

        normed_w = self._spectral_normalizer(self.w, is_training=is_training)

        outputs = tf.nn.convolution(
            tensor,
            normed_w,
            strides=self.stride,
            padding=self.conv_padding,
            dilations=self.rate,
            data_format=self.data_format)

        if self.with_bias:
            outputs = tf.nn.bias_add(
                outputs, self.b, data_format=self.data_format)

        return outputs


class Linear(snt.Module):
    """Simple linear layer.

    Linear map from [batch_size, input_size] -> [batch_size, output_size].
    """

    def __init__(self, output_size):
        """Constructor."""
        super().__init__(name=None)
        self._output_size = output_size
        self._linear = snt.Linear(output_size=output_size)

    def __call__(self, tensor):
        return self._linear(tensor)


class BatchNorm(snt.Module):
    """Batch normalization."""

    def __init__(self, calc_sigma=True):
        """Constructor."""
        super().__init__(name=None)
        self._calc_sigma = calc_sigma
        self._batch_norm = snt.BatchNorm(
            create_scale=calc_sigma, create_offset=True)

    def __call__(self, tensor, is_training=True):
        return self._batch_norm(tensor, is_training=is_training)


class ApplyAlongAxis(snt.Module):
    """Layer for applying an operation on each element, along a specified axis."""

    def __init__(self, operation, axis=0):
        """Constructor."""
        super().__init__(name=None)
        self._operation = operation
        self._axis = axis

    def __call__(self, *args):
        """Apply the operation to each element of args along the specified axis."""
        split_inputs = [tf.unstack(arg, axis=self._axis) for arg in args]
        res = [self._operation(x) for x in zip(*split_inputs)]
        return tf.stack(res, axis=self._axis)


class ApplyAlongAxis2(snt.Module):
    """Layer for applying an operation on each element, along a specified axis."""

    def __init__(self, operation, axis=0):
        """Constructor."""
        super().__init__(name=None)
        self._operation = operation
        self._axis = axis

    def __call__(self, inputs):
        """Apply the operation to each element of args along the specified axis."""
        split_inputs = tf.unstack(inputs, axis=self._axis)
        res = [self._operation(x) for x in split_inputs]
        return tf.stack(res, axis=self._axis)


class SpectralNormalizer(snt.Module):

    def __init__(self, epsilon=1e-12, name=None):
        super().__init__(name=name)
        self.l2_normalize = functools.partial(
            tf.math.l2_normalize, epsilon=epsilon)

    @snt.once
    def _initialize(self, weights):
        init = self.l2_normalize(snt.initializers.TruncatedNormal()(
            shape=[1, weights.shape[-1]], dtype=weights.dtype))
        # 'u' tracks our estimate of the first spectral vector for the given weight.
        self.u = tf.Variable(init, name='u', trainable=False)

    def __call__(self, weights, is_training=True):
        self._initialize(weights)
        if is_training:
            # Do a power iteration and update u and weights.
            weights_matrix = tf.reshape(weights, [-1, weights.shape[-1]])
            v = self.l2_normalize(self.u @ tf.transpose(weights_matrix))
            v_w = v @ weights_matrix
            u = self.l2_normalize(v_w)
            sigma = tf.stop_gradient(tf.reshape(v_w @ tf.transpose(u), []))
            self.u.assign(u)
            weights.assign(weights / sigma)
        return weights
