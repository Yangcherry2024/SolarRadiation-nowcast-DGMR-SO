# import sonnet as snt
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
import numpy as np
import time
import datetime


class DGMR(tf.keras.Model):
    def __init__(self, lead_time=240, time_delta=15) -> None:
        super(DGMR, self).__init__()
        self.strategy = None
        self.global_step = 0
        self.generator_obj = Generator(
            lead_time=lead_time, time_delta=time_delta)
        self.discriminator_obj = Discriminator()

    @tf.function
    def __call__(self, tensor, is_training=False):
        return self.generator_obj(tensor, is_training=is_training)

    def compile(self, gen_optimizer, disc_optimizer, gen_loss, disc_loss):
        # super(DGMR, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss

    def fit(self, dataset_aug, data_val, steps=2, callbacks=[]):
        train_writer = callbacks[0]
        ckpt_manager = callbacks[1]
        ckpt = callbacks[2]
        # tf.profiler.experimental.start(callbacks[3])

        disc_loss_l = []
        gen_loss_l = []

        '''if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')'''
        dataset_aug = iter(dataset_aug)
        dataset_val = iter(data_val)
        num_batches = self.global_step

        for step in range(steps):
            # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):


            tf.print("step is: " + str(step))
            batch_inputs1, batch_targets1, targ_mask1 = next(dataset_aug)
            batch_targets1 = batch_targets1[:, :, :, :, :]
            batch_inputs2, batch_targets2, targ_mask2 = next(dataset_aug)
            batch_targets2 = batch_targets2[:, :, :, :, :]
            batch_inputs3, batch_targets3, targ_mask3 = next(dataset_aug)
            batch_targets3 = batch_targets3[:, :, :, :, :]
            batch_inputs4, batch_targets4, targ_mask4 = next(dataset_aug)
            batch_targets4 = batch_targets4[:, :, :, :, :]

            # the size of images need to be changed to (224,128), in order to march the model
            batch_inputs1 = tf.pad(batch_inputs1, [[0, 0], [0, 0], [
                0, 16], [0, 2], [0, 0]], mode='CONSTANT')
            batch_targets1 = tf.pad(batch_targets1, [[0, 0], [0, 0], [
                0, 16], [0, 2], [0, 0]], mode='CONSTANT')
            batch_inputs2 = tf.pad(batch_inputs2, [[0, 0], [0, 0], [
                0, 16], [0, 2], [0, 0]], mode='CONSTANT')
            batch_targets2 = tf.pad(batch_targets2, [[0, 0], [0, 0], [
                0, 16], [0, 2], [0, 0]], mode='CONSTANT')
            batch_inputs3 = tf.pad(batch_inputs3, [[0, 0], [0, 0], [
                0, 16], [0, 2], [0, 0]], mode='CONSTANT')
            batch_targets3 = tf.pad(batch_targets3, [[0, 0], [0, 0], [
                0, 16], [0, 2], [0, 0]], mode='CONSTANT')
            batch_inputs4 = tf.pad(batch_inputs4, [[0, 0], [0, 0], [
                0, 16], [0, 2], [0, 0]], mode='CONSTANT')
            batch_targets4 = tf.pad(batch_targets4, [[0, 0], [0, 0], [
                0, 16], [0, 2], [0, 0]], mode='CONSTANT')

            temp_time = time.time()

            gen_loss, disc_loss = self.train_step(
                batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2, batch_inputs3,
                batch_targets3, targ_mask3, batch_inputs4, batch_targets4, targ_mask4)

            tf.print("Debugging: before run")
            # Loss
            disc_loss_l.append(disc_loss)
            gen_loss_l.append(gen_loss)

            if step and (step % 200 == 0):
                val_input1, val_target1, label1 = next(dataset_val)
                val_input2, val_target2, label2 = next(dataset_val)

                val_input = tf.pad(val_input1, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
                val_target = tf.pad(val_target1, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
                input = tf.pad(val_input2, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
                target = tf.pad(val_target2, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')

                val_gen_loss, val_disc_loss = self.val_step(
                    val_input, val_target, label1, input, target, label2)
                tf.print("val_gen_loss", val_gen_loss,
                         "val_disc_loss", val_disc_loss)

                with train_writer.as_default():
                    tf.summary.scalar("val_gen_loss", val_gen_loss, step=step)
                    tf.summary.scalar("val_disc_loss", val_disc_loss, step=step)

                if (step % 5000 == 0):
                    targets_1,input1, target_8, input8, target_16, input16, obv_img, pred_img = self.data_process(val_input[:1],
                                                                                                val_target[:1])
                    obv_img = np.squeeze(obv_img,axis=(0))
                    pred_img = np.squeeze(pred_img,axis=(0))
                    with train_writer.as_default():
                        tf.summary.image("observed image", obv_img, max_outputs=16, step=step)
                        tf.summary.image("predicted image", pred_img, max_outputs=16, step=step)

            if step and (step % 3000 == 0):
                # dataset_val = data_val.take(200)
                dataset_val_eva = iter(data_val)

                rmse_1 = []
                r_1 = []
                ssim_1 = []
                rmse_8 = []
                r_8 = []
                ssim_8 = []
                rmse_16 = []
                r_16 = []
                ssim_16 = []

                for i in range(180):
                    val_input1, val_target1, label1 = next(dataset_val_eva)
                    val_target1 = val_target1[:, :, :, :, :]
                    val_input = tf.pad(val_input1, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')
                    val_target = tf.pad(val_target1, [[0, 0], [0, 0], [0, 16], [0, 2], [0, 0]], mode='CONSTANT')

                    targets_1,input1, target_8, input8, target_16, input16, obv_img, pred_img = self.data_process(val_input[:],
                                                                                                val_target[:])
                    if len(target_8) == 0 or len(input8) == 0 or len(target_16) == 0 or len(input16) == 0 or len(input1) == 0 or len(targets_1) == 0:
                        continue
                    r_metric_val_1 = self.r_evaluation(input1, targets_1)
                    rmse_metric_val_1 = self.rmse_evaluation(input1, targets_1)
                    ssim_metric_1 = tf.image.ssim(obv_img[:, 0:1, :, :, :], pred_img[:, 0:1, :, :, :], 1)
                    
                    r_metric_val_8 = self.r_evaluation(input8, target_8)
                    rmse_metric_val_8 = self.rmse_evaluation(input8, target_8)
                    ssim_metric_8 = tf.image.ssim(obv_img[:, 7:8, :, :, :], pred_img[:, 7:8, :, :, :], 1)

                    r_metric_val_16 = self.r_evaluation(input16, target_16)
                    rmse_metric_val_16 = self.rmse_evaluation(input16, target_16)
                    ssim_metric_16 = tf.image.ssim(obv_img[:, 15:, :, :, :], pred_img[:, 15:, :, :, :], 1)
                    
                    r_1 = np.append(r_1, r_metric_val_1)
                    rmse_1 = np.append(rmse_1, rmse_metric_val_1)
                    ssim_1 = np.append(ssim_1, ssim_metric_1)

                    r_8 = np.append(r_8, r_metric_val_8)
                    rmse_8 = np.append(rmse_8, rmse_metric_val_8)
                    ssim_8 = np.append(ssim_8, ssim_metric_8)

                    r_16 = np.append(r_16, r_metric_val_16)
                    rmse_16 = np.append(rmse_16, rmse_metric_val_16)
                    ssim_16 = np.append(ssim_16, ssim_metric_16)
                    
                r_1 = self.cal_mean(r_1)
                rmse_1 = self.cal_mean(rmse_1)
                ssim_1 = self.cal_mean(ssim_1)
                r_8 = self.cal_mean(r_8)
                rmse_8 = self.cal_mean(rmse_8)
                ssim_8 = self.cal_mean(ssim_8)
                r_16 = self.cal_mean(r_16)
                rmse_16 = self.cal_mean(rmse_16)
                ssim_16 = self.cal_mean(ssim_16)

                with train_writer.as_default():
                    tf.summary.scalar("r_score_val_1", r_1, step=step)
                    tf.summary.scalar("rmse_score_val_1", rmse_1, step=step)
                    tf.summary.scalar("ssim_score_val_1", ssim_1, step=step)
                    tf.summary.scalar("r_score_val_8", r_8, step=step)
                    tf.summary.scalar("rmse_score_val_8", rmse_8, step=step)
                    tf.summary.scalar("ssim_score_val_8", ssim_8, step=step)
                    tf.summary.scalar("r_score_val_16", r_16, step=step)
                    tf.summary.scalar("rmse_score_val_16", rmse_16, step=step)
                    tf.summary.scalar("ssim_score_val_16", ssim_16, step=step)

            num_batches += 1
            if num_batches < 500:
                tf.print("Time per epoch: ", time.time() - temp_time)

            if step and (step % 100 == 0):
                tf.print("Gen Loss: ", gen_loss.numpy(),
                         " Disc Loss: ", disc_loss.numpy())
                # val loss
                # TODO evaluate with evaluation metrics

            if step and (step % 2000 == 0):
                ckpt_save_path = ckpt_manager.save()

            if step and (step % 5 == 0):
                with train_writer.as_default():
                    tf.summary.scalar("Gen_loss", gen_loss, step=step)
                    tf.summary.scalar("Disc_loss", disc_loss, step=step)

        # tf.profiler.experimental.stop()

        return gen_loss_l, disc_loss_l

    # (input_signature=[tf.TensorSpec(shape=[2, 4, 256, 256, 1], dtype=tf.float32), tf.TensorSpec(shape=[2, 2, 256, 256, 1], dtype=tf.float32)])

    def random_crop_images(self,target_data, label_data, crop_height, crop_width):
        target_shape = tf.shape(target_data)
        # 随机生成剪裁起始位置
        target_y = tf.random.uniform(shape=[], maxval=target_shape[1] - crop_height + 1, dtype=tf.int32)
        target_x = tf.random.uniform(shape=[], maxval=target_shape[2] - crop_width + 1, dtype=tf.int32)
        label_y = target_y
        label_x = target_x
        # 在target_data上进行剪裁
        target_cropped = tf.image.crop_to_bounding_box(target_data, target_y, target_x, crop_height, crop_width)
        # 在label_data上进行剪裁
        label_cropped = tf.image.crop_to_bounding_box(label_data, label_y, label_x, crop_height, crop_width)
        return target_cropped, label_cropped

    # @tf.function
    def distributed_train_step(self, batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2,
                               targ_mask2):
        tf.print("Debugging: before run")
        per_replica_g_losses, per_replica_d_losses = self.strategy.run(self.train_step,
                                                                       args=(batch_inputs1, batch_targets1, targ_mask1,
                                                                             batch_inputs2, batch_targets2, targ_mask2))
        tf.print("Debugging: after run")

        tf.print("Debugging: per_replica_g_losses -> ", per_replica_g_losses)
        tf.print("Debugging: per_replica_d_losses -> ", per_replica_d_losses)

        total_g_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_g_losses, axis=0)
        total_d_loss = self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica_d_losses, axis=0)

        tf.print("Debugging: total_g_loss -> ", total_g_loss)
        tf.print("Debugging: total_d_loss -> ", total_d_loss)
        return total_g_loss, total_d_loss

    def cal_mean(self, array):
        filtered = [x for x in array if x is not None and not np.isnan(x)]
        array_mean = np.mean(filtered)
        return array_mean

    def r_evaluation(self, batch_inputs2, batch_targets2):
        # batch_targets2,predict_labels = self.data_process(batch_inputs2,batch_targets2)
        r_metric_train = np.corrcoef(batch_targets2, batch_inputs2)[0, 1]
        return r_metric_train

    def mae_evaluation(self, batch_inputs2, batch_targets2):
        # batch_targets2,predict_labels = self.data_process(batch_inputs2,batch_targets2)
        error = np.mean(np.abs(batch_inputs2 - batch_targets2))
        return error

    def rmse_evaluation(self, obs, sim):
        obs = obs.flatten()
        sim = sim.flatten()
        # obs, sim = self.maskarray(obs, sim)

        return np.sqrt(np.mean((obs - sim) ** 2))

    def data_process(self, batch_inputs2, batch_targets2):
        predict_labels = self.generator_obj(batch_inputs2)
        predict_labels = predict_labels.numpy()
        
        predict_labels_16 = predict_labels[:, 15:, :, :, :]
        predict_labels_16 = np.squeeze(predict_labels_16)

        batch_targets2 = batch_targets2.numpy()
        batch_targets2_16 = batch_targets2[:, 15:, :, :, :]
        batch_targets2_16 = np.squeeze(batch_targets2_16)
        predict_labels_16 = predict_labels_16.flatten()
        batch_targets2_16 = batch_targets2_16.flatten()

        mask_obs_16 = np.where(batch_targets2_16 <= 0, 0, 1)
        mask_sim_16 = np.where(predict_labels_16 <= 0, 0, 1)
        mask_16 = np.where((mask_obs_16 + mask_sim_16) <= 1, 0, 1).astype(bool)
        batch_targets2_16 = batch_targets2_16[mask_16]
        predict_labels_16 = predict_labels_16[mask_16]

        predict_labels_8 = predict_labels[:, 7:8, :, :, :]
        predict_labels_8 = np.squeeze(predict_labels_8)

        batch_targets2_8 = batch_targets2[:, 7:8, :, :, :]
        batch_targets2_8 = np.squeeze(batch_targets2_8)
        predict_labels_8 = predict_labels_8.flatten()
        batch_targets2_8 = batch_targets2_8.flatten()

        mask_obs_8 = np.where(batch_targets2_8 <= 0, 0, 1)
        mask_sim_8 = np.where(predict_labels_8 <= 0, 0, 1)
        mask_8 = np.where((mask_obs_8 + mask_sim_8) <= 1, 0, 1).astype(bool)
        batch_targets2_8 = batch_targets2_8[mask_8]
        predict_labels_8 = predict_labels_8[mask_8]
        
        predict_labels_1 = predict_labels[:, 0:1, :, :, :]
        predict_labels_1 = np.squeeze(predict_labels_1)

        batch_targets2_1 = batch_targets2[:, 0:1, :, :, :]
        batch_targets2_1 = np.squeeze(batch_targets2_1)
        predict_labels_1 = predict_labels_1.flatten()
        batch_targets2_1 = batch_targets2_1.flatten()

        mask_obs_1 = np.where(batch_targets2_1 <= 0, 0, 1)
        mask_sim_1 = np.where(predict_labels_1 <= 0, 0, 1)
        mask_1 = np.where((mask_obs_1 + mask_sim_1) <= 1, 0, 1).astype(bool)
        batch_targets2_1 = batch_targets2_1[mask_1]
        predict_labels_1 = predict_labels_1[mask_1]
        return batch_targets2_1,predict_labels_1,batch_targets2_8, predict_labels_8, batch_targets2_16, predict_labels_16, batch_targets2, predict_labels

    @tf.function
    def val_step(self, batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2):

        self.val_disc_step(batch_inputs1, batch_targets1, targ_mask1)
        self.val_disc_step(batch_inputs2, batch_targets2, targ_mask2)

        self.val_disc_step(batch_inputs1, batch_targets1, targ_mask1)
        disc_loss = self.val_disc_step(batch_inputs2, batch_targets2, targ_mask2)

        gen_loss = self.val_gen_step(batch_inputs2, batch_targets2, targ_mask2)
        return gen_loss, disc_loss

    def val_disc_step(self, batch_inputs, batch_targets, targ_mask, is_training=True):
        batch_predictions = self.generator_obj(
            batch_inputs, is_training=is_training)
        '''batch_predictions = tf.where(
            tf.equal(targ_mask, True), batch_predictions, -1)'''
        gen_sequence = tf.concat(
            [batch_inputs[..., :1], batch_predictions], axis=1)
        real_sequence = tf.concat(
            [batch_inputs[..., :1], batch_targets], axis=1)
        # gen_sequence = tf.cast(gen_sequence, tf.double)
        # real_sequence = tf.cast(real_sequence, tf.double)
        concat_inputs = tf.concat(
            [real_sequence, gen_sequence], axis=0)

        concat_outputs = self.discriminator_obj(
            concat_inputs, is_training=is_training)

        score_real, score_generated = tf.split(
            concat_outputs, 2, axis=0)
        disc_loss = self.disc_loss(score_generated, score_real)
        return disc_loss

    def val_gen_step(self, batch_inputs, batch_targets, targ_mask, is_training=True):
        num_samples_per_input = 1  # FIXME it was 6.
        gen_samples = [self.generator_obj(batch_inputs, is_training=is_training)
                       for _ in range(num_samples_per_input)]

        grid_cell_reg = grid_cell_regularizer(tf.stack(gen_samples, axis=0),
                                              batch_targets)

        gen_sequences = [tf.concat([batch_inputs[..., :1], x], axis=1)
                         for x in gen_samples]
        real_sequence = tf.concat(
            [batch_inputs[..., :1], batch_targets], axis=1)

        generated_scores = []
        for g_seq in gen_sequences:
            concat_inputs = tf.concat([real_sequence, g_seq], axis=0)
            concat_outputs = self.discriminator_obj(
                concat_inputs, is_training=is_training)
            score_real, score_generated = tf.split(
                concat_outputs, 2, axis=0)
            generated_scores.append(score_generated)
        gen_disc_loss = self.gen_loss(tf.concat(generated_scores, axis=0))
        gen_loss = gen_disc_loss + 1 * grid_cell_reg
        return gen_loss

    @tf.function
    def train_step(self, batch_inputs1, batch_targets1, targ_mask1, batch_inputs2, batch_targets2, targ_mask2,
                   batch_inputs3, batch_targets3, targ_mask3, batch_inputs4, batch_targets4, targ_mask4):
        self.disc_step(batch_inputs1, batch_targets1, targ_mask1)
        self.disc_step(batch_inputs2, batch_targets2, targ_mask2)
        self.disc_step(batch_inputs3, batch_targets3, targ_mask3)
        disc_loss = self.disc_step(batch_inputs4, batch_targets4, targ_mask4)

        gen_loss = self.gen_step(batch_inputs4, batch_targets4, targ_mask4)

        tf.print("Debugging: total_g_loss -> ", gen_loss)
        tf.print("Debugging: total_d_loss -> ", disc_loss)
        return gen_loss, disc_loss

    def disc_step(self, batch_inputs, batch_targets, targ_mask, is_training=True):
        tf.print("Debugging: disc_step: ", "Disc step has started",
                 str(datetime.datetime.now()))
        with tf.GradientTape() as disc_tape:
            batch_predictions = self.generator_obj(
                batch_inputs, is_training=is_training)
            '''batch_predictions = tf.where(
                tf.equal(targ_mask, True), batch_predictions, -1)'''
            gen_sequence = tf.concat(
                [batch_inputs[..., :1], batch_predictions], axis=1)
            real_sequence = tf.concat(
                [batch_inputs[..., :1], batch_targets], axis=1)
            # gen_sequence = tf.cast(gen_sequence, tf.double)
            # real_sequence = tf.cast(real_sequence, tf.double)
            concat_inputs = tf.concat(
                [real_sequence, gen_sequence], axis=0)

            concat_outputs = self.discriminator_obj(
                concat_inputs, is_training=is_training)

            score_real, score_generated = tf.split(
                concat_outputs, 2, axis=0)
            disc_loss = self.disc_loss(score_generated, score_real)

        disc_grads = disc_tape.gradient(
            disc_loss, self.discriminator_obj.trainable_variables)

        # Aggregate the gradients from the full batch.
        '''replica_ctx_disc = tf.distribute.get_replica_context()
        disc_grads = replica_ctx_disc.all_reduce(
            tf.distribute.ReduceOp.MEAN, disc_grads)'''

        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator_obj.trainable_variables))

        tf.print("Debugging: disc_step: ", "Disc step ended",
                 str(datetime.datetime.now()))
        return disc_loss

    def gen_step(self, batch_inputs, batch_targets, targ_mask, is_training=True):
        tf.print("Debugging: gen_step: ", "Gen step has started",
                 str(datetime.datetime.now()))
        with tf.GradientTape() as gen_tape:
            num_samples_per_input = 1  # FIXME it was 6.
            gen_samples = [self.generator_obj(batch_inputs, is_training=is_training)
                           for _ in range(num_samples_per_input)]

            grid_cell_reg = grid_cell_regularizer(tf.stack(gen_samples, axis=0),
                                                  batch_targets)
            '''gen_samples = [tf.where(
                tf.equal(targ_mask, True), gen_sample, -1) for gen_sample in gen_samples]'''
            gen_sequences = [tf.concat([batch_inputs[..., :1], x], axis=1)
                             for x in gen_samples]
            real_sequence = tf.concat(
                [batch_inputs[..., :1], batch_targets], axis=1)

            generated_scores = []
            for g_seq in gen_sequences:
                # real_sequence = tf.cast(real_sequence,tf.double)
                # g_seq = tf.cast(g_seq, tf.double)
                concat_inputs = tf.concat([real_sequence, g_seq], axis=0)
                concat_outputs = self.discriminator_obj(
                    concat_inputs, is_training=is_training)
                score_real, score_generated = tf.split(
                    concat_outputs, 2, axis=0)
                generated_scores.append(score_generated)

            gen_disc_loss = self.gen_loss(tf.concat(generated_scores, axis=0))
            gen_loss = gen_disc_loss + 1 * grid_cell_reg

        gen_grads = gen_tape.gradient(
            gen_loss, self.generator_obj.trainable_variables)

        # Aggregate the gradients from the full batch.
        '''replica_ctx_disc = tf.distribute.get_replica_context()
        disc_grads = replica_ctx_disc.all_reduce(
            tf.distribute.ReduceOp.MEAN, disc_grads)'''

        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.generator_obj.trainable_variables))

        # tf.print("Debugging: gen_loss -> ", gen_loss)

        # tf.print("Debugging: gen_step: ", "Gen step has ended", str(datetime.datetime.now()))
        return gen_loss


def grid_cell_regularizer(generated_samples, batch_targets):
    """Grid cell regularizer.

    Args:
      generated_samples: Tensor of size [n_samples, batch_size, 18, 256, 256, 1].
      batch_targets: Tensor of size [batch_size, 18, 256, 256, 1].

    Returns:
      loss: A tensor of shape [batch_size].
    """
    gen_mean = tf.reduce_mean(generated_samples, axis=0)
    weights = tf.clip_by_value(batch_targets, 0.0, 0.1)
    loss = tf.reduce_mean(tf.math.abs(gen_mean - batch_targets)*weights)
    return loss
