import functools
import discriminator
import latent_stack
import layers
import tensorflow.compat.v1 as tf
import sonnet as snt


class Generator(snt.Module):
    """Generator for the proposed model."""

    def __init__(self, lead_time=240, time_delta=15):
        tf.print("Debugging: Generator initialization has started")
        """Constructor.

        Args:
          lead_time: last lead time for the generator to predict. Default: 90 min.
          time_delta: time step between predictions. Default: 5 min.
        """
        super().__init__(name=None)
        #self.prec_normalizer = tf.keras.layers.Normalization(axis=None)
        #self.feature_normalizer = tf.keras.layers.Normalization(axis=None)
        
        
        # self._batch_size=batch_size
        #self.twoStream_3DConvNet = TwoStream_3DConvNet() #FIXME comment to Yang: Removed that
        self._cond_stack = ConditioningStack()
        self._sampler = Sampler(lead_time, time_delta)

        tf.print("Debugging: Generator has been initialized")

    def __call__(self, inputs, is_training=True):
        """Connect to a graph.

        Args:
          inputs: a batch of inputs on the shape [batch_size, time, h, w, 1].
        Returns:
          predictions: a batch of predictions in the form
            [batch_size, num_lead_times, h, w, 1].
        """
        
        #inputs = tf.stack((self.prec_normalizer(inputs[..., :1]), self.feature_normalizer(inputs[..., 1:3])), axis=-1)
        tf.print("call generator")
        #input_representation = self.twoStream_3DConvNet(
        #    inputs, is_training=is_training)
        _, _, height, width, _ = inputs.shape.as_list()
        initial_states = self._cond_stack(
            inputs, is_training=is_training)
        predictions = self._sampler(
            initial_states, [height, width], is_training=is_training)
        return predictions

    def get_variables(self):
        """Get all variables of the module."""
        pass


"""class TwoStream_3DConvNet(snt.Module): #FIXME comment to Yang: Removed that
    def __init__(self,):
        tf.print("Debugging: Generator initialization has started")
        '''Constructor.

        '''
        super().__init__(name=None)

        self.Unit3D = layers.SNConv3D
        self._bn1 = layers.BatchNorm

        self.conv3D_refl = layers.SNConv3D(output_channels=1, kernel_size=3)
        self._bn1 = layers.BatchNorm(calc_sigma=False)
        '''self.conv3D_echo = layers.SNConv3D(output_channels=1, kernel_size=3)
        self._bn2 = layers.BatchNorm(calc_sigma=False)'''
        self.conv3D_mot = layers.SNConv3D(output_channels=1, kernel_size=3)
        self._bn3 = layers.BatchNorm(calc_sigma=False)
        self.conv3D_fuse = layers.SNConv3D(output_channels=1, kernel_size=3)
        self._bn4 = layers.BatchNorm(calc_sigma=False)

    def __call__(self, tensor, is_training=True):
        #tensor_list = tf.unstack(tensor, axis=-1)
        stream1 = self.conv3D_refl(tensor[..., :1], is_training=is_training)
        #stream1 = tf.nn.relu(stream1)
        #stream1 = self.Unit3D(output_channels=4, kernel_size=3)(stream1)
        stream1 = tf.nn.relu(self._bn1(stream1, is_training=is_training))

        #stream2 = self.conv3D_echo(tensor[..., 1:2], is_training=is_training)
        #stream2 = tf.nn.relu(stream2)
        #stream2 = self.Unit3D(output_channels=4, kernel_size=3)(stream2)
        #stream2 = tf.nn.relu(self._bn2(stream2, is_training=is_training))

        # stream3 = self.conv3D_mot(tensor[..., 1:3], is_training=is_training)
        #stream3 = tf.nn.relu(stream3)
        #stream3 = self.Unit3D(output_channels=4, kernel_size=3)(stream3)
        # stream3 = tf.nn.relu(self._bn3(stream3, is_training=is_training))

        # stream = self.conv3D_fuse(
        #     tf.concat([stream1, stream3], axis=-1), is_training=is_training)
        stream = self.conv3D_fuse(
              stream1, is_training=is_training)
        stream = tf.nn.relu(
            self._bn4(stream, is_training=is_training))  # stream3

        return stream"""


class ConditioningStack(snt.Module):
    """Conditioning Stack for the Generator."""

    def __init__(self):
        super().__init__(name=None)
        self._block1 = discriminator.DBlock(
            output_channels=48, downsample=True)
        self._conv_mix1 = layers.SNConv2D(output_channels=48, kernel_size=3)
        self._block2 = discriminator.DBlock(
            output_channels=96, downsample=True)
        self._conv_mix2 = layers.SNConv2D(output_channels=96, kernel_size=3)
        self._block3 = discriminator.DBlock(
            output_channels=192, downsample=True)
        self._conv_mix3 = layers.SNConv2D(output_channels=192, kernel_size=3)
        self._block4 = discriminator.DBlock(
            output_channels=384, downsample=True)
        self._conv_mix4 = layers.SNConv2D(output_channels=384, kernel_size=3)
        #self._batch_size = batch_size

    def __call__(self, inputs, is_training=True):
        # Space to depth conversion of 256x256x1 radar to 128x128x4 hiddens.
        h0 = batch_apply(
            functools.partial(tf.nn.space_to_depth, block_size=2), inputs)

        # Downsampling residual D Blocks.
        h1 = time_apply(self._block1, h0)
        h2 = time_apply(self._block2, h1)
        h3 = time_apply(self._block3, h2)
        h4 = time_apply(self._block4, h3)

        # Spectrally normalized convolutions, followed by rectified linear units.
        init_state_1 = self._mixing_layer(
            h1, self._conv_mix1, is_training=is_training)
        init_state_2 = self._mixing_layer(
            h2, self._conv_mix2, is_training=is_training)
        init_state_3 = self._mixing_layer(
            h3, self._conv_mix3, is_training=is_training)
        init_state_4 = self._mixing_layer(
            h4, self._conv_mix4, is_training=is_training)

        # Return a stack of conditioning representations of size 64x64x48, 32x32x96,
        # 16x16x192 and 8x8x384.
        return init_state_1, init_state_2, init_state_3, init_state_4

    def _mixing_layer(self, inputs, conv_block, is_training):
        # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
        # then perform convolution on the output while preserving number of c.
        stacked_inputs = tf.concat(tf.unstack(inputs, axis=1), axis=-1)
        return tf.nn.relu(conv_block(stacked_inputs, is_training=is_training))

# sampler makes predictions of 16 future radiation images
class Sampler(snt.Module):
    """Sampler for the Generator."""

    def __init__(self, lead_time=90, time_delta=5):
        super().__init__(name=None)
        self._num_predictions = lead_time // time_delta
        # self._batch_size=batch_size
        self._latent_stack = latent_stack.LatentCondStack()

        self._conv_gru4 = ConvGRU(num_channels=384)
        self._conv4 = layers.SNConv2D(kernel_size=1, output_channels=768)
        self._gblock4 = GBlock(output_channels=768)
        self._g_up_block4 = UpsampleGBlock(output_channels=384)

        self._conv_gru3 = ConvGRU(num_channels=192)
        self._conv3 = layers.SNConv2D(kernel_size=1, output_channels=384)
        self._gblock3 = GBlock(output_channels=384)
        self._g_up_block3 = UpsampleGBlock(output_channels=192)

        self._conv_gru2 = ConvGRU(num_channels=96)
        self._conv2 = layers.SNConv2D(kernel_size=1, output_channels=192)
        self._gblock2 = GBlock(output_channels=192)
        self._g_up_block2 = UpsampleGBlock(output_channels=96)

        self._conv_gru1 = ConvGRU(num_channels=48)
        self._conv1 = layers.SNConv2D(kernel_size=1, output_channels=96)
        self._gblock1 = GBlock(output_channels=96)
        self._g_up_block1 = UpsampleGBlock(output_channels=48)

        self._bn = layers.BatchNorm()
        self._output_conv = layers.SNConv2D(kernel_size=1, output_channels=4)

    def __call__(self, initial_states, resolution, is_training=True):
        init_state_1, init_state_2, init_state_3, init_state_4 = initial_states
        batch_size = init_state_1.shape.as_list()[0]

        # Latent conditioning stack.
        z = self._latent_stack(batch_size, resolution)
        hs = [z] * self._num_predictions

        # Layer 4 (bottom-most).
        hs = tf.stack(hs)
        hs, _ = snt.static_unroll(self._conv_gru4, hs, init_state_4)
        hs = tf.unstack(hs)
        hs = [self._conv4(h, is_training=is_training) for h in hs]
        hs = [self._gblock4(h, is_training=is_training) for h in hs]
        hs = [self._g_up_block4(h, is_training=is_training) for h in hs]

        # Layer 3.
        hs = tf.stack(hs)
        # init_state_3 = init_state_3[:, :-1, :-1, :]
        hs, _ = snt.static_unroll(self._conv_gru3, hs, init_state_3)
        hs = tf.unstack(hs)
        hs = [self._conv3(h, is_training=is_training) for h in hs]
        hs = [self._gblock3(h, is_training=is_training) for h in hs]
        hs = [self._g_up_block3(h, is_training=is_training) for h in hs]

        # Layer 2.
        hs = tf.stack(hs)
        hs, _ = snt.static_unroll(self._conv_gru2, hs, init_state_2)
        hs = tf.unstack(hs)
        hs = [self._conv2(h, is_training=is_training) for h in hs]
        hs = [self._gblock2(h, is_training=is_training) for h in hs]
        hs = [self._g_up_block2(h, is_training=is_training) for h in hs]

        # Layer 1 (top-most).
        hs = tf.stack(hs)
        hs, _ = snt.static_unroll(self._conv_gru1, hs, init_state_1)
        hs = tf.unstack(hs)
        hs = [self._conv1(h, is_training=is_training) for h in hs]
        hs = [self._gblock1(h, is_training=is_training) for h in hs]
        hs = [self._g_up_block1(h, is_training=is_training) for h in hs]

        # Output layer.
        hs = [tf.nn.relu(self._bn(h, is_training=is_training)) for h in hs]
        hs = [self._output_conv(h, is_training=is_training) for h in hs]
        hs = [tf.nn.depth_to_space(h, 2) for h in hs]

        return tf.stack(hs, axis=1)


class GBlock(snt.Module):
    """Residual generator block without upsampling."""

    def __init__(self, output_channels, sn_eps=0.0001):
        super().__init__(name=None)
        self._conv1_3x3 = layers.SNConv2D(
            output_channels, kernel_size=3, sn_eps=sn_eps)
        self._bn1 = layers.BatchNorm()
        self._conv2_3x3 = layers.SNConv2D(
            output_channels, kernel_size=3, sn_eps=sn_eps)
        self._bn2 = layers.BatchNorm()
        self.conv_1x1 = layers.SNConv2D(
            output_channels, kernel_size=1, sn_eps=sn_eps)
        self._output_channels = output_channels
        self._sn_eps = sn_eps

    def __call__(self, inputs, is_training=True):
        input_channels = inputs.shape[-1]

        # Optional spectrally normalized 1x1 convolution.
        if input_channels != self._output_channels:
            sc = self.conv_1x1(inputs, is_training=is_training)
        else:
            sc = inputs

        # Two-layer residual connection, with batch normalization, nonlinearity and
        # 3x3 spectrally normalized convolution in each layer.
        h = tf.nn.relu(self._bn1(inputs, is_training=is_training))
        h = self._conv1_3x3(h, is_training=is_training)
        h = tf.nn.relu(self._bn2(h, is_training=is_training))
        h = self._conv2_3x3(h, is_training=is_training)

        # Residual connection.
        return h + sc


class UpsampleGBlock(snt.Module):
    """Upsampling residual generator block."""

    def __init__(self, output_channels, sn_eps=0.0001):
        super().__init__(name=None)
        self._conv_1x1 = layers.SNConv2D(
            output_channels, kernel_size=1, sn_eps=sn_eps)
        self._conv1_3x3 = layers.SNConv2D(
            output_channels, kernel_size=3, sn_eps=sn_eps)
        self._bn1 = layers.BatchNorm()
        self._conv2_3x3 = layers.SNConv2D(
            output_channels, kernel_size=3, sn_eps=sn_eps)
        self._bn2 = layers.BatchNorm()
        self._output_channels = output_channels

    def __call__(self, inputs, is_training=True):
        # x2 upsampling and spectrally normalized 1x1 convolution.
        sc = layers.upsample_nearest_neighbor(inputs, upsample_size=2)
        sc = self._conv_1x1(sc, is_training=is_training)

        # Two-layer residual connection, with batch normalization, nonlinearity and
        # 3x3 spectrally normalized convolution in each layer, and x2 upsampling in
        # the first layer.
        h = tf.nn.relu(self._bn1(inputs, is_training=is_training))
        h = layers.upsample_nearest_neighbor(h, upsample_size=2)
        h = self._conv1_3x3(h, is_training=is_training)
        h = tf.nn.relu(self._bn2(h, is_training=is_training))
        h = self._conv2_3x3(h, is_training=is_training)

        # Residual connection.
        return h + sc


class ConvGRU(snt.Module):
    """A ConvGRU implementation."""

    def __init__(self, kernel_size=3, sn_eps=0.0001, num_channels=768):
        """Constructor.

        Args:
          kernel_size: kernel size of the convolutions. Default: 3.
          sn_eps: constant for spectral normalization. Default: 1e-4.
        """
        super().__init__()
        self._kernel_size = kernel_size
        self._sn_eps = sn_eps
        self._conv1 = layers.SNConv2D(
            num_channels, self._kernel_size, sn_eps=self._sn_eps)
        self._conv2 = layers.SNConv2D(
            num_channels, self._kernel_size, sn_eps=self._sn_eps)
        self._conv3 = layers.SNConv2D(
            num_channels, self._kernel_size, sn_eps=self._sn_eps)

    def __call__(self, inputs, prev_state, is_training=True):
        # self._initialize(inputs)
        # self._initialize(inputs)
        # Concatenate the inputs and previous state along the channel axis.
        num_channels = prev_state.shape[-1]
        xh = tf.concat([inputs, prev_state], axis=-1)

        # Read gate of the GRU.
        # read_gate_conv = layers.SNConv2D(
        #    num_channels, self._kernel_size, sn_eps=self._sn_eps)
        read_gate = tf.math.sigmoid(self._conv1(xh, is_training=is_training))

        # Update gate of the GRU.
        # update_gate_conv = layers.SNConv2D(
        #    num_channels, self._kernel_size, sn_eps=self._sn_eps)
        update_gate = tf.math.sigmoid(self._conv2(xh, is_training=is_training))

        # Gate the inputs.
        gated_input = tf.concat([inputs, read_gate * prev_state], axis=-1)

        # Gate the cell and state / outputs.
        # output_conv = layers.SNConv2D(
        #    num_channels, self._kernel_size, sn_eps=self._sn_eps)
        c = tf.nn.relu(self._conv3(gated_input, is_training=is_training))
        out = update_gate * prev_state + (1. - update_gate) * c
        new_state = out
        return out, new_state


def time_apply(func, inputs):
    """Apply function func on each element of inputs along the time axis."""
    return layers.ApplyAlongAxis2(func, axis=1)(inputs)


def batch_apply(func, inputs):
    """Apply function func on each element of inputs along the batch axis."""
    return layers.ApplyAlongAxis2(func, axis=0)(inputs)
    # return tf.map_fn(func, inputs)
