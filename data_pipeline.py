import tensorflow as tf
from pathlib import Path

def slicing(x0, x1, x2, x3):
    num_conditioning_frames = 4
    lead_frames = 18
    return (tf.concat([x0[..., :1], x0[..., 3:5]], axis=-1), x1[:lead_frames, ...], x2[:num_conditioning_frames + lead_frames, ...],x3)


def masking(x0, x1, x2, x3):
    num_conditioning_frames = 4
    x0_mask = x2[:num_conditioning_frames, ...]
    x1_mask = x2[num_conditioning_frames:, ...]

    # x0_scaled = tf.stack((scaled, x0[...,1:]), axis=-1)
    # tf.print(x0_scaled.shape)
    x0 = tf.where(tf.equal(x0_mask, True), x0, -1 / 32)
    # x1 = tf.where(tf.equal(x1_mask, True), x1, -1/32)

    return (x0, x1, x1_mask, x3)

def parse_tfr_element_simple(element):
    data = {
        'window_cond': tf.io.FixedLenFeature([], tf.float32),
        'height_cond': tf.io.FixedLenFeature([], tf.float32),
        'width_cond': tf.io.FixedLenFeature([], tf.float32),
        'raw_image_cond': tf.io.FixedLenFeature([], tf.string),
        'depth_cond': tf.io.FixedLenFeature([], tf.float32),
        'window_targ': tf.io.FixedLenFeature([], tf.float32),
        'height_targ': tf.io.FixedLenFeature([], tf.float32),
        'width_targ': tf.io.FixedLenFeature([], tf.float32),
        'raw_image_targ': tf.io.FixedLenFeature([], tf.string),
        'depth_targ': tf.io.FixedLenFeature([], tf.float32),
        'window_mask': tf.io.FixedLenFeature([], tf.int64),
        'height_mask': tf.io.FixedLenFeature([], tf.int64),
        'width_mask': tf.io.FixedLenFeature([], tf.int64),
        'raw_image_mask': tf.io.FixedLenFeature([], tf.string),
        'depth_mask': tf.io.FixedLenFeature([], tf.int64),
        'start_date': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    window_cond = content['window_cond']
    height_cond = content['height_cond']
    width_cond = content['width_cond']
    depth_cond = content['depth_cond']
    raw_image_cond = content['raw_image_cond']
    window_targ = content['window_targ']
    height_targ = content['height_targ']
    width_targ = content['width_targ']
    depth_targ = content['depth_targ']
    raw_image_targ = content['raw_image_targ']
    window_mask = content['window_mask']
    height_mask = content['height_mask']
    width_mask = content['width_mask']
    depth_mask = content['depth_mask']
    raw_image_mask = content['raw_image_mask']
    start_date = content['start_date']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature_cond = tf.io.parse_tensor(raw_image_cond, out_type=tf.float32)
    feature_cond = tf.reshape(
        feature_cond, shape=[window_cond, height_cond, width_cond, depth_cond])

    feature_targ = tf.io.parse_tensor(raw_image_targ, out_type=tf.float32)
    feature_targ = tf.reshape(
        feature_targ, shape=[window_targ, height_targ, width_targ, depth_targ])

    feature_mask = tf.io.parse_tensor(raw_image_mask, out_type=tf.bool)
    feature_mask = tf.reshape(
        feature_mask, shape=[window_mask, height_mask, width_mask, depth_mask])

    feature_date = tf.io.parse_tensor(start_date, out_type=tf.string)
    return (feature_cond, feature_targ, feature_mask, feature_date)  # , feature_date


def Dataset(tfr_dir,  batch_size, prob=False, shuffle=True):
    files = tf.io.gfile.glob(str(tfr_dir)+'/*.tfrecords')
    files = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        #files = files.shuffle(buffer_size=len(files)//10)
        files = files.shuffle(buffer_size=len(files) // 1)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
                               num_parallel_calls=tf.data.AUTOTUNE)
    if prob:
        dataset = dataset.map(parse_tfr_element_with_prob_simple,
                              num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.map(parse_tfr_element_simple,
                              num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(slicing,
                          num_parallel_calls=tf.data.AUTOTUNE)
    '''dataset = dataset.map(reflectivity_to_rainrate,
                          num_parallel_calls=tf.data.AUTOTUNE)'''
    dataset = dataset.map(masking,
                          num_parallel_calls=tf.data.AUTOTUNE)
    # FIXME increase for better shuffling
    if shuffle:
        dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE).repeat()
    return dataset

train_data = Dataset(Path('Data'), batch_size=5)





def reflectivity_to_rainrate(x0, x1, x2,x3):
    Z = (x0 * 0.5) - 32.0
    rain_rate = ((10 ** (Z/10)) / 200) ** 0.625
    return (rain_rate, x1, x2,x3)

def parse_tfr_element_with_prob_simple(element):
    data = {
        'window_cond': tf.io.FixedLenFeature([], tf.float32),
        'height_cond': tf.io.FixedLenFeature([], tf.float32),
        'width_cond': tf.io.FixedLenFeature([], tf.float32),
        'raw_image_cond': tf.io.FixedLenFeature([], tf.string),
        'depth_cond': tf.io.FixedLenFeature([], tf.float32),
        'window_targ': tf.io.FixedLenFeature([], tf.float32),
        'height_targ': tf.io.FixedLenFeature([], tf.float32),
        'width_targ': tf.io.FixedLenFeature([], tf.float32),
        'raw_image_targ': tf.io.FixedLenFeature([], tf.string),
        'depth_targ': tf.io.FixedLenFeature([], tf.float32),
        'window_mask': tf.io.FixedLenFeature([], tf.int64),
        'height_mask': tf.io.FixedLenFeature([], tf.int64),
        'width_mask': tf.io.FixedLenFeature([], tf.int64),
        'raw_image_mask': tf.io.FixedLenFeature([], tf.string),
        'depth_mask': tf.io.FixedLenFeature([], tf.int64),
        'prob': tf.io.FixedLenFeature([], tf.string),
        'start_date': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    window_cond = content['window_cond']
    height_cond = content['height_cond']
    width_cond = content['width_cond']
    depth_cond = content['depth_cond']
    raw_image_cond = content['raw_image_cond']
    window_targ = content['window_targ']
    height_targ = content['height_targ']
    width_targ = content['width_targ']
    depth_targ = content['depth_targ']
    raw_image_targ = content['raw_image_targ']
    window_mask = content['window_mask']
    height_mask = content['height_mask']
    width_mask = content['width_mask']
    depth_mask = content['depth_mask']
    raw_image_mask = content['raw_image_mask']
    prob = content['prob']
    start_date = content['start_date']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature_cond = tf.io.parse_tensor(raw_image_cond, out_type=tf.float32)
    feature_cond = tf.reshape(
        feature_cond, shape=[window_cond, height_cond, width_cond, depth_cond])

    feature_targ = tf.io.parse_tensor(raw_image_targ, out_type=tf.float32)
    feature_targ = tf.reshape(
        feature_targ, shape=[window_targ, height_targ, width_targ, depth_targ])

    feature_mask = tf.io.parse_tensor(raw_image_mask, out_type=tf.bool)
    feature_mask = tf.reshape(
        feature_mask, shape=[window_mask, height_mask, width_mask, depth_mask])

    feature_prob = tf.io.parse_tensor(prob, out_type=tf.float32)

    feature_date = tf.io.parse_tensor(start_date, out_type=tf.string)
    # , feature_prob, feature_date
    return (feature_cond, feature_targ, feature_mask, feature_date)



