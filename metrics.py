import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt
import pysteps
import numpy as np
from pysteps.utils.spectral import rapsd
from pysteps.utils import rapsd, transformation
from pysteps.verification.probscores import CRPS

class CSI_score():
    def __init__(self, threshold) -> None:
        self.truepositives = tf.keras.metrics.TruePositives()
        self.falsepositives = tf.keras.metrics.FalsePositives()
        self.falsenegative = tf.keras.metrics.FalseNegatives()
        self.threshold = threshold

    def __call__(self, batch_gen, batch_target):
        pred = crop_middle(batch_gen)
        targ = crop_middle(batch_target)
        pred = pred > self.threshold
        targ = targ > self.threshold
        batch_gen_u = tf.unstack(pred, axis=1)
        batch_target_u = tf.unstack(targ, axis=1)

        csi_score = [((self.truepositives(pred_temp, targ_temp) + 1) / ((self.truepositives(pred_temp, targ_temp) + self.falsepositives
                      (pred_temp, targ_temp) + self.falsenegative(pred_temp, targ_temp)) + 1)).numpy() for pred_temp, targ_temp in zip(batch_gen_u, batch_target_u)]
        return np.array(csi_score)


class BIAS_score():
    def __init__(self, threshold) -> None:
        self.truepositives = tf.keras.metrics.TruePositives()
        self.falsepositives = tf.keras.metrics.FalsePositives()
        self.falsenegative = tf.keras.metrics.FalseNegatives()
        self.threshold = threshold

    def __call__(self, batch_gen, batch_target):
        pred = crop_middle(batch_gen)
        targ = crop_middle(batch_target)
        pred = pred > self.threshold
        targ = targ > self.threshold
        batch_gen_u = tf.unstack(pred, axis=1)
        batch_target_u = tf.unstack(targ, axis=1)

        bias_score = [(((self.truepositives(pred_temp, targ_temp) + self.falsepositives(pred_temp, targ_temp)) + 1) / ((self.truepositives
                      (pred_temp, targ_temp) + self.falsenegative(pred_temp, targ_temp)) + 1)).numpy() for pred_temp, targ_temp in zip(batch_gen_u, batch_target_u)]

        return np.array(bias_score)


class MSE_score():
    def __init__(self) -> None:
        pass

    def __call__(self, batch_gen, batch_target):
        pred = crop_middle(batch_gen)
        targ = crop_middle(batch_target)
        pred_u = tf.unstack(pred, axis=1)
        targ_u = tf.unstack(targ, axis=1)

        mse_score = [tf.reduce_mean(tf.math.square(pred - targ)).numpy()
                     for pred, targ in zip(pred_u, targ_u)]
        return np.array(mse_score)

class MSE_score1():
    def __init__(self) -> None:
        pass

    def __call__(self, batch_gen, batch_target):
        pred = tf.pad(batch_gen, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
        targ = tf.pad(batch_target, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
        pred_u = tf.unstack(pred, axis=1)
        targ_u = tf.unstack(targ, axis=1)

        # Compute the mean of each tensor
        mean1 = tf.reduce_mean(batch_gen)
        mean2 = tf.reduce_mean(batch_target)

        # Compute the centered tensors
        centered_tensor1 = batch_target - mean1
        centered_tensor2 = batch_target - mean2

        # Compute the correlation coefficient
        for centered_tensor1, centered_tensor2 in zip(pred_u, targ_u):
            r_score = tf.reduce_sum(centered_tensor1 * centered_tensor2) / (
                    tf.sqrt(tf.reduce_sum(tf.square(centered_tensor1))) *
                    tf.sqrt(tf.reduce_sum(tf.square(centered_tensor2)))
            )
        return np.array(r_score)


class Rank_histogram():
    def __init__(self) -> None:
        pass

    def __call__(self, batch_gen_list, batch_target):
        pred_l = [crop_middle(pred) for pred in batch_gen_list]
        pred = tf.stack(pred_l, axis=0)
        targ = crop_middle(batch_target)

        tf.print("pred", pred.shape)
        tf.print("targ", targ.shape)

        rank_scores = np.zeros((pred.shape[0] + 1)).astype('float32')
        counter = 0
        for i in range(targ.shape[0]):
            for j in range(targ.shape[1]):
                rank_scores += pysteps.verification.ensscores.rankhist(
                    pred[:, i, j, ..., 0].numpy(), targ[i, j, ..., 0].numpy())
                counter += 1
        rank_score = rank_scores/counter
        return np.array(rank_score)


class Max_Pool_CRPS():
    def __init__(self, K) -> None:
        T = 18  # FIXME change that to 18
        stride = K // 4
        if stride == 0:
            stride = 1
        self.max_pool_3d = tf.keras.layers.MaxPool3D(
            pool_size=(1, K, K), strides=(1, stride, stride), padding='Valid')

    def __call__(self, b_pred, b_targ):
        pred = [crop_middle(pred) for pred in b_pred]
        targ = crop_middle(b_targ)

        #rand_ind = np.random.randint(low=0, high=len(pred), size=(2))
        pooled_pred = [self.max_pool_3d(pre) for pre in pred]
        pooled_targ = self.max_pool_3d(targ)

        pooled_pred = tf.stack(pooled_pred, axis=1)
        pooled_crps_score = [CRPS(pooled_pred.numpy()[0, :, i, ..., 0],
                                  pooled_targ.numpy()[0, i, ..., 0]) for i in range(pooled_pred.shape[2])]
        # tf.reduce_mean(np.abs(pooled_pred[rand_ind[0]] - pooled_targ) - 0.5 * np.abs(
        # pooled_pred[rand_ind[0]] - pooled_pred[rand_ind[1]]), axis=[0, 2, 3, 4])

        return np.array(pooled_crps_score)


class Avg_Pool_CRPS():
    def __init__(self, K) -> None:
        T = 18  # FIXME change that to 20
        stride = K // 4
        if stride == 0:
            stride = 1
        self.avg_pool_3d = tf.keras.layers.AvgPool3D(
            pool_size=(1, K, K), strides=(1, stride, stride), padding='Valid')

    def __call__(self, b_pred, b_targ):
        pred = [crop_middle(pred) for pred in b_pred]
        targ = crop_middle(b_targ)

        pooled_pred = [self.avg_pool_3d(pre) for pre in pred]
        pooled_targ = self.avg_pool_3d(targ)

        pooled_pred = tf.stack(pooled_pred, axis=1)
        pooled_crps_score = [CRPS(pooled_pred.numpy()[0, :, i, ..., 0],
                                  pooled_targ.numpy()[0, i, ..., 0]) for i in range(pooled_pred.shape[2])]

        return np.array(pooled_crps_score)


class PSD():
    def __init__(self, timestep) -> None:
        T = 18  # FIXME change that to 20
        self.timestep = timestep

    def __call__(self, b_pred):
        pred = crop_middle(b_pred[:, self.timestep: self.timestep+1, ...])

        # Log-transform the data
        R, metadata = transformation.dB_transform(
            pred.numpy()[0, 0, ..., 0], threshold=0.1, zerovalue=-15.0)

        # Assign the fill value to all the Nans
        R[~np.isfinite(R)] = metadata["zerovalue"]
        R_, freq = rapsd(R, fft_method=np.fft, return_freq=True)

        return (R_, freq)
