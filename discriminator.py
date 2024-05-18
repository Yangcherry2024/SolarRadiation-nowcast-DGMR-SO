import layers
import tensorflow.compat.v1 as tf
import sonnet as snt


class Discriminator(snt.Module):
    """Discriminator."""

    def __init__(self):
        #tf.print("Debugging: Discriminator initialization has started")
        """Constructor."""
        super().__init__(name=None)
        # FIXME replace everything that is changed
        # Number of random time steps for the spatial discriminator.
        self._num_spatial_frames = 8
        # Input size ratio with respect to crop size for the temporal discriminator.
        self._temporal_crop_ratio = 2
        # As the input is the whole sequence of the event (including conditioning
        # frames), the spatial discriminator needs to pick only the t > T+0.
        self._num_conditioning_frames = 4
        #self._batch_size = batch_size
        self._spatial_discriminator = SpatialDiscriminator()
        self._temporal_discriminator = TemporalDiscriminator()

        #tf.print("Debugging: Discriminator has been initialized")

    def __call__(self, frames, is_training=True):
        """Build the discriminator.

        Args:
          frames: a tensor with a complete observation [b, 22, 256, 256, 1].

        Returns:
          A tensor with discriminator loss scalars [b, 2].
        """
        b, t, h, w, c = frames.shape.as_list()
        # Prepare the frames for spatial discriminator: pick 8 random time steps out
        # of 18 lead time steps, and downsample from 256x256 to 128x128.
        target_frames_sel = tf.range(self._num_conditioning_frames, t)
        permutation = tf.stack([
            tf.random.shuffle(target_frames_sel)[:self._num_spatial_frames]
            for _ in range(b)
        ], 0)
        frames_for_sd = tf.gather(frames, permutation, batch_dims=1)
        frames_for_sd = tf.layers.AveragePooling3D(
            [1, 2, 2], [1, 2, 2], data_format='channels_last')(frames_for_sd)

        # Compute the average spatial discriminator score for each of 8 picked time
        # steps.
        sd_out = self._spatial_discriminator(
            frames_for_sd, is_training=is_training)

        # Prepare the frames for temporal discriminator: choose the offset of a
        # random crop of size 128x128 out of 256x256 and pick full sequence samples.
        cr = self._temporal_crop_ratio
        h_offset = tf.random_uniform([], 0, (cr - 1) * (h // cr), tf.int32)
        w_offset = tf.random_uniform([], 0, (cr - 1) * (w // cr), tf.int32)
        zero_offset = tf.zeros_like(w_offset)
        begin_tensor = tf.stack(
            [zero_offset, zero_offset, h_offset, w_offset, zero_offset], -1)
        size_tensor = tf.constant([b, t, h // cr, w // cr, c])
        frames_for_td = tf.slice(frames, begin_tensor, size_tensor)
        frames_for_td.set_shape([b, t, h // cr, w // cr, c])

        # Compute the average temporal discriminator score over length 5 sequences.
        td_out = self._temporal_discriminator(
            frames_for_td, is_training=is_training)

        return tf.concat([sd_out, td_out], 1)


class DBlock(snt.Module):
    """Convolutional residual block."""

    def __init__(self, output_channels, kernel_size=3, downsample=True,
                 pre_activation=True, conv=layers.SNConv2D,
                 pooling=layers.downsample_avg_pool, activation=tf.nn.relu):
        """Constructor for the D blocks of the DVD-GAN.

        Args:
          output_channels: Integer number of channels in the second convolution, and
            number of channels in the residual 1x1 convolution module.
          kernel_size: Integer kernel size of the convolutions.
          downsample: Boolean: shall we use the average pooling layer?
          pre_activation: Boolean: shall we apply pre-activation to inputs?
          conv: TF module, either layers.Conv2D or a wrapper with spectral
            normalisation layers.SNConv2D.
          pooling: Average pooling layer. Default: layers.downsample_avg_pool.
          activation: Activation at optional preactivation and first conv layers.
        """
        super().__init__(name=None)
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._downsample = downsample
        self._pre_activation = pre_activation
        self._conv1 = conv(output_channels=self._output_channels,
                           kernel_size=self._kernel_size)
        self._conv2 = conv(output_channels=self._output_channels,
                           kernel_size=self._kernel_size)
        self._residual_conv = conv(output_channels=self._output_channels,
                                   kernel_size=1)
        self._pooling = pooling
        self._activation = activation

    def __call__(self, inputs, is_training=True):
        """Build the DBlock.

        Args:
          inputs: a tensor with a complete observation [b, 256, 256, 1]

        Returns:
          A tensor with discriminator loss scalars [b].
        """
        h0 = inputs

        # Pre-activation.
        if self._pre_activation:
            h0 = self._activation(h0)

        # First convolution.
        input_channels = h0.shape.as_list()[-1]
        h1 = self._conv1(h0, is_training=is_training)
        h1 = self._activation(h1)

        # Second convolution.
        h2 = self._conv2(h1, is_training=is_training)

        # Downsampling.
        if self._downsample:
            h2 = self._pooling(h2)

        # The residual connection, make sure it has the same dimensionality
        # with additional 1x1 convolution and downsampling if needed.
        if input_channels != self._output_channels or self._downsample:
            sc = self._residual_conv(inputs, is_training=is_training)
            if self._downsample:
                sc = self._pooling(sc)
        else:
            sc = inputs

        # Residual connection.
        return h2 + sc


class SpatialDiscriminator(snt.Module):
    """Spatial Discriminator."""

    def __init__(self):
        super().__init__(name=None)
        self._block1 = DBlock(output_channels=48, pre_activation=False)
        self._block2 = DBlock(output_channels=96)
        self._block3 = DBlock(output_channels=192)
        self._block4 = DBlock(output_channels=384)
        self._block5 = DBlock(output_channels=768)
        self._block6 = DBlock(output_channels=768, downsample=False)

        self._bn = layers.BatchNorm(calc_sigma=False)
        self._linear = layers.Linear(output_size=1)

        # self._batch_size=batch_size

    def __call__(self, frames, is_training=True):
        """Build the spatial discriminator.

        Args:
          frames: a tensor with a complete observation [b, n, 128, 128, 1].

        Returns:
          A tensor with discriminator loss scalars [b].
        """
        b, n, h, w, c = frames.shape.as_list()
        # Process each of the n inputs independently.
        frames = tf.reshape(frames, [b * n, h, w, c])

        # Space-to-depth stacking from 128x128x1 to 64x64x4.
        frames = tf.nn.space_to_depth(frames, block_size=2)

        # Five residual D Blocks to halve the resolution of the image and double
        # the number of channels.
        y = self._block1(frames, is_training=is_training)
        y = self._block2(y, is_training=is_training)
        y = self._block3(y, is_training=is_training)
        y = self._block4(y, is_training=is_training)
        y = self._block5(y, is_training=is_training)

        # One more D Block without downsampling or increase in number of channels.
        y = self._block6(y, is_training=is_training)

        # Sum-pool the representations and feed to spectrally normalized lin. layer.
        y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
        y = self._bn(y, is_training=is_training)
        output = self._linear(y)

        # Take the sum across the t samples. Note: we apply the ReLU to
        # (1 - score_real) and (1 + score_generated) in the loss.
        output = tf.reshape(output, [b, n, 1])
        output = tf.reduce_sum(output, keepdims=True, axis=1)
        return output


class TemporalDiscriminator(snt.Module):
    """Spatial Discriminator."""

    def __init__(self):
        super().__init__(name=None)
        self._block1 = DBlock(output_channels=48, conv=layers.SNConv3D,
                              pooling=layers.downsample_avg_pool3d,
                              pre_activation=False)
        self._block2 = DBlock(output_channels=96, conv=layers.SNConv3D,
                              pooling=layers.downsample_avg_pool3d)
        self._block3 = DBlock(output_channels=192)
        self._block4 = DBlock(output_channels=384)
        self._block5 = DBlock(output_channels=768)
        self._block6 = DBlock(output_channels=768, downsample=False)
        self._bn = layers.BatchNorm(calc_sigma=False)
        self._linear = layers.Linear(output_size=1)
        # self._batch_size=batch_size

    def __call__(self, frames, is_training=True):
        """Build the temporal discriminator.

        Args:
          frames: a tensor with a complete observation [b, ts, 128, 128, 1]

        Returns:
          A tensor with discriminator loss scalars [b].
        """
        b, ts, hs, ws, cs = frames.shape.as_list()
        # Process each of the ti inputs independently.
        frames = tf.reshape(frames, [b * ts, hs, ws, cs])

        # Space-to-depth stacking from 128x128x1 to 64x64x4.
        frames = tf.nn.space_to_depth(frames, block_size=2)

        # Stack back to sequences of length ti.
        frames = tf.reshape(frames, [b, ts, hs, ws, cs])

        # Two residual 3D Blocks to halve the resolution of the image, double
        # the number of channels, and reduce the number of time steps.
        y = self._block1(frames, is_training=is_training)
        y = self._block2(y, is_training=is_training)

        # Get t < ts, h, w, and c, as we have downsampled in 3D.
        _, t, h, w, c = y.shape.as_list()

        # Process each of the t images independently.
        # b t h w c -> (b x t) h w c
        y = tf.reshape(y, [-1] + [h, w, c])

        # Three residual D Blocks to halve the resolution of the image and double
        # the number of channels.
        y = self._block3(y, is_training=is_training)
        y = self._block4(y, is_training=is_training)
        y = self._block5(y, is_training=is_training)

        # One more D Block without downsampling or increase in number of channels.
        y = self._block6(y, is_training=is_training)

        # Sum-pool the representations and feed to spectrally normalized lin. layer.
        y = tf.reduce_sum(tf.nn.relu(y), axis=[1, 2])
        y = self._bn(y, is_training=is_training)
        output = self._linear(y)

        # Take the sum across the t samples. Note: we apply the ReLU to
        # (1 - score_real) and (1 + score_generated) in the loss.
        output = tf.reshape(output, [b, t, 1])
        scores = tf.reduce_sum(output, keepdims=True, axis=1)
        return scores
