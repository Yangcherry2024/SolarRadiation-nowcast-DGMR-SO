# import tensorflow.compat.v1 as tf
import tensorflow as tf
from keras.optimizers import Adam
from data_pipeline_single import Dataset
from pathlib import Path
from dgmr import DGMR
from losses import Loss_hing_disc, Loss_hing_gen
import os
import matplotlib.pyplot as plt
from utils import *
from tensorflow.python import debug as tfdbg
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print('GPU:', tf.config.list_physical_devices('GPU'))

#try:
    # Disable all GPUS
#    tf.config.set_visible_devices([], 'GPU')
#    visible_devices = tf.config.get_visible_devices()
#    for device in visible_devices:
#        assert device.device_type != 'GPU'
#except:
    # Invalid device or cannot modify virtual devices once initialized.
#    pass

# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
cfg = read_yaml(Path('/data1/cui/progrom/DGMR_new1/configs/' + 'train_0' + '.yml'))

MODEL_NAME = cfg['model_identification']['model_name']
MODEL_VERSION = cfg['model_identification']['model_version']
ROOT = get_project_root()
CHECKPOINT_DIR = ROOT / 'Checkpoints' / \
    (str(MODEL_NAME) + '_v' + str(MODEL_VERSION))
make_dirs([CHECKPOINT_DIR])

training_steps = cfg['model_params']['steps']

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(18)
# tf.config.set_soft_device_placement(True)

# gpu_devices_list = tf.config.list_physical_devices('GPU')

batch_size = 16
# the size of images need to be changed to (224,128), in order to march the model
train_data,train_dataset_aug = Dataset(Path('/data1/cui/data/train_sds_single_day_opt_csc_20_21/'), batch_size=batch_size)
val_data,val_data_val = Dataset(Path('/data1/cui/data/train_sds_single_day_opt_csc_20_21/'), batch_size=batch_size)

train_writer = tf.summary.create_file_writer(
    str(ROOT / "logs" / (str(MODEL_NAME) + '_v' + str(MODEL_VERSION)) / "train/"))

prof_dir = str(ROOT / "logs" / (str(MODEL_NAME) +
                                '_v' + str(MODEL_VERSION)) / "profiler/")
# profiler_writer = tf.summary.create_file_writer(prof_dir)

# INIT MODEL
disc_optimizer = Adam(learning_rate=2E-4, beta_1=0.0, beta_2=0.999)
gen_optimizer = Adam(learning_rate=1E-5, beta_1=0.0, beta_2=0.999)
loss_hinge_gen = Loss_hing_gen()
loss_hinge_disc = Loss_hing_disc()

# with strategy.scope() :
my_model = DGMR(lead_time=240, time_delta=15)
my_model.trainable = True
my_model.compile(gen_optimizer, disc_optimizer,
                 loss_hinge_gen, loss_hinge_disc)

# my_model.strategy = strategy

ckpt = tf.train.Checkpoint(generator=my_model.generator_obj,
                           discriminator=my_model.discriminator_obj,
                           generator_optimizer=my_model.gen_optimizer,
                           discriminator_optimizer=my_model.disc_optimizer)
ckpt_manager = tf.train.CheckpointManager(
    ckpt, CHECKPOINT_DIR, max_to_keep=100)

if ckpt_manager.latest_checkpoint:
    #ckpt.restore(ckpt_manager.latest_checkpoint)
    ckpt.restore(ckpt_manager.checkpoints[-45])
    print('Latest checkpoint restored!!')


# train_dist_dataset = strategy.experimental_distribute_dataset(train_data)

gen_loss, disc_loss = my_model.fit(train_dataset_aug, val_data, steps=training_steps, callbacks=[
    train_writer, ckpt_manager, ckpt, prof_dir])  # , prof_dir
