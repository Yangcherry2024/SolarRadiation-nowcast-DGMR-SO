from pathlib import Path
import yaml
import os
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import os
#from pysteps.visualization import plot_spectrum1d


def get_project_root() -> Path:
    return Path(__file__).parent


def read_yaml(file_path) -> str:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def split_data_xy_1(data):
    x = data[0:4, :, :, :]
    y = data[4:20,:, :, :]
    return x, y

def split_data_xy_multi(data):
    x = data[0:4, :, :, :]
    y = data[4:20,:, :, -1]
    return x, y

def split_time_xy_1(time):
    x = time[0:4]
    y = time[4:20]
    return x, y


def split_data_xy(data):
    x = data[:,0:4, :, :, :]
    y = data[:,4:20,:, :, :]
    return x, y

def split_time_xy(time):
    x = time[:,0:4]
    y = time[:,4:20]
    return x, y

def make_dirs(list_dir: list) -> None:
    for l in list_dir:
        l.mkdir(parents=True, exist_ok=True)


def read_date(date):
    date = date.numpy()[0].decode('utf-8')
    decoded_date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    return decoded_date


def date_to_name(date):
    decoded_date = date.strftime('%Y-%m-%d-%H-%M-%S')
    return decoded_date

def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size >> 20


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def crop_middle(tensor):
    # Prepare the frames for temporal discriminator: choose the offset of a
    # random crop of size 128x128 out of 256x256 and pick full sequence samples.
    b, t, h, w, c = tensor.shape
    cr = 2
    h_offset = (h // cr) // 2
    w_offset = (w // cr) // 2
    zero_offset = tf.zeros_like(w_offset)
    begin_tensor = tf.stack(
        [zero_offset, zero_offset, h_offset, w_offset, zero_offset], -1)
    size_tensor = tf.constant([b, t, h // cr, w // cr, c])
    frames_for_eval = tf.slice(tensor, begin_tensor, size_tensor)
    frames_for_eval.set_shape([b, t, h // cr, w // cr, c])
    return frames_for_eval


def crop_middle_ensemble(tensor):
    # Prepare the frames for temporal discriminator: choose the offset of a
    b, e, t, h, w, c = tensor.shape
    cr = 2
    h_offset = (h // cr) // 2
    w_offset = (w // cr) // 2
    zero_offset = tf.zeros_like(w_offset)
    begin_tensor = tf.stack(
        [zero_offset, zero_offset, zero_offset, h_offset, w_offset, zero_offset], -1)
    size_tensor = tf.constant([b, e, t, h // cr, w // cr, c])
    frames = tf.slice(tensor, begin_tensor, size_tensor)
    frames.set_shape([b, e, t, h // cr, w // cr, c])
    return frames


def plot_csi_score(score1, score4, score8):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,
                           sharey=True, figsize=(1, 3))

    fig.text(0.5, 0., 'Prediction interval [min]', ha='center')
    fig.text(0.08, 0.5, 'CSI', va='center', rotation='vertical')
    fig.subplots_adjust(bottom=0.15)

    fig.set_figheight(2.5)
    fig.set_figwidth(11)
    #axis.set_ylabel('common xlabel')

    time_ax = len(score1) * 5
    ax[0].plot(range(0, time_ax, 5), score1)
    ax[0].set_title('Precipitation [mm/h] > 1.0')
    #ax[0].set_ylim(0, .8)
    ax[0].set_xlim(0, time_ax)

    ax[1].plot(range(0, time_ax, 5), score4)
    ax[1].set_title('Precipitation [mm/h] > 4.0')
    #ax[1].set_ylim(0, .8)
    ax[1].set_xlim(0, time_ax)

    ax[2].plot(range(0, time_ax, 5), score8)
    ax[2].set_title('Precipitation [mm/h] > 8.0')
    #ax[2].set_ylim(0, .8)
    ax[2].set_xlim(0, time_ax)

    return plot_to_image(fig)


def plot_bias_score(score1, score4, score8):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,
                           sharey=True, figsize=(1, 3))

    fig.text(0.5, 0., 'Prediction interval [min]', ha='center')
    fig.text(0.08, 0.5, 'BIAS', va='center', rotation='vertical')
    fig.subplots_adjust(bottom=0.15)

    fig.set_figheight(2.5)
    fig.set_figwidth(11)
    #axis.set_ylabel('common xlabel')

    time_ax = len(score1) * 5
    ax[0].plot(range(0, time_ax, 5), score1)
    ax[0].set_title('Precipitation [mm/h] > 1.0')
    #ax[0].set_ylim(0, 2.)
    ax[0].set_xlim(0, time_ax)

    ax[1].plot(range(0, time_ax, 5), score4)
    ax[1].set_title('Precipitation [mm/h] > 4.0')
    #ax[1].set_ylim(0, 2.)
    ax[1].set_xlim(0, time_ax)

    ax[2].plot(range(0, time_ax, 5), score8)
    ax[2].set_title('Precipitation [mm/h] > 8.0')
    #ax[2].set_ylim(0, 2.)
    ax[2].set_xlim(0, time_ax)

    return plot_to_image(fig)


def plot_mse_score(score):
    fig = plt.figure()
    plt.xlabel('Prediction interval[min]')
    plt.ylabel('MSE')
    plt.plot(range(0, len(score) * 5, 5), score)
    #plt.ylim([0, 0.5])

    return plot_to_image(fig)


def plot_rankhist_score(score):
    fig = plt.figure()
    plt.title('Rank Histogram')
    plt.xlabel('Rank')
    plt.ylabel('Proportion of samples')
    plt.plot(range(len(score)), score)
    #plt.ylim([0, 0.6])

    return plot_to_image(fig)


def plot_max_pool_crps_score(score1, score4, score16):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,
                           sharey=True, figsize=(1, 3))

    fig.text(0.5, 0., 'Prediction interval [min]', ha='center')
    fig.text(0.08, 0.5, 'Max-Pooled CRPS', va='center', rotation='vertical')
    fig.subplots_adjust(bottom=0.15)

    fig.set_figheight(2.5)
    fig.set_figwidth(11)

    time_ax = len(score1) * 5
    ax[0].plot(range(0, time_ax, 5), score1)
    ax[0].set_title('Precipitation [mm/h] > 1.0')
    #ax[0].set_ylim(0, 1.)
    ax[0].set_xlim(0, time_ax)

    ax[1].plot(range(0, time_ax, 5), score4)
    ax[1].set_title('Precipitation [mm/h] > 4.0')
    #ax[1].set_ylim(0, 1.)
    ax[1].set_xlim(0, time_ax)

    ax[2].plot(range(0, time_ax, 5), score16)
    ax[2].set_title('Precipitation [mm/h] > 16.0')
    #ax[2].set_ylim(0, 1.)
    ax[2].set_xlim(0, time_ax)

    return plot_to_image(fig)


def plot_avg_pool_crps_score(score1, score4, score16):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,
                           sharey=True, figsize=(1, 3))

    fig.text(0.5, 0., 'Prediction interval [min]', ha='center')
    fig.text(0.08, 0.5, 'Avg-Pooled CRPS', va='center', rotation='vertical')
    fig.subplots_adjust(bottom=0.15)

    fig.set_figheight(2.5)
    fig.set_figwidth(11)

    time_ax = len(score1) * 5
    ax[0].plot(range(0, time_ax, 5), score1)
    ax[0].set_title('Precipitation [mm/h] > 1.0')
    #ax[0].set_ylim(0, 1.)
    ax[0].set_xlim(0, time_ax)

    ax[1].plot(range(0, time_ax, 5), score4)
    ax[1].set_title('Precipitation [mm/h] > 4.0')
    #ax[1].set_ylim(0, 1.)
    ax[1].set_xlim(0, time_ax)

    ax[2].plot(range(0, time_ax, 5), score16)
    ax[2].set_title('Precipitation [mm/h] > 16.0')
    #ax[2].set_ylim(0, 1.)
    ax[2].set_xlim(0, time_ax)

    return plot_to_image(fig)


def plot_avg_pool_crps_score(score1, score4, score16):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,
                           sharey=True, figsize=(1, 3))

    fig.text(0.5, 0., 'Prediction interval [min]', ha='center')
    fig.text(0.08, 0.5, 'Avg-Pooled CRPS', va='center', rotation='vertical')
    fig.subplots_adjust(bottom=0.15)

    fig.set_figheight(2.5)
    fig.set_figwidth(11)

    time_ax = len(score1) * 5
    ax[0].plot(range(0, time_ax, 5), score1)
    ax[0].set_title('Precipitation [mm/h] > 1.0')
    #ax[0].set_ylim(0, 1.)
    ax[0].set_xlim(0, time_ax)

    ax[1].plot(range(0, time_ax, 5), score4)
    ax[1].set_title('Precipitation [mm/h] > 4.0')
    #ax[1].set_ylim(0, 1.)
    ax[1].set_xlim(0, time_ax)

    ax[2].plot(range(0, time_ax, 5), score16)
    ax[2].set_title('Precipitation [mm/h] > 16.0')
    #ax[2].set_ylim(0, 1.)
    ax[2].set_xlim(0, time_ax)

    return plot_to_image(fig)


# def psd_score(score30, score60, score90, freqs):
#     fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True,
#                            sharey=True, figsize=(1, 3))
#
#     #fig.text(0.5, 0., 'Prediction interval [min]', ha='center')
#     fig.text(0.08, 0.5, 'PSD', va='center', rotation='vertical')
#     fig.subplots_adjust(bottom=0.15)
#
#     fig.set_figheight(2.5)
#     fig.set_figwidth(11)
#
#     plot_scales = [1024, 256, 64, 16, 4]
#     plot_spectrum1d(
#         freqs,
#         score30,
#         x_units="km",
#         # y_units="PSD",
#         color="k",
#         ax=ax[0],
#         # label="Observed",
#         wavelength_ticks=plot_scales,
#     )
#     #ax[0].set_ylim(-20, 50)
#     ax[0].set_ylabel("")
#
#     plot_spectrum1d(
#         freqs,
#         score60,
#         x_units="km",
#         # y_units="PSD",
#         color="k",
#         ax=ax[1],
#         # label="Observed",
#         wavelength_ticks=plot_scales,
#     )
#     #ax[1].set_ylim(-40, 60)
#     ax[1].set_ylabel("")
#
#     plot_spectrum1d(
#         freqs,
#         score90,
#         x_units="km",
#         # y_units="PSD",
#         color="k",
#         ax=ax[2],
#         # label="Observed",
#         wavelength_ticks=plot_scales,
#     )
#     #ax[2].set_ylim(-20, 50)
#     ax[2].set_ylabel("")
#
#     return plot_to_image(fig)
