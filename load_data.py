# -*- coding: utf-8 -*-
import tensorflow as tf

def parse_function(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([1024, 2], tf.float32),
        'y': tf.io.FixedLenFeature([8], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [1024, 2])
    y = tf.reshape(parsed_example['y'], [8])
    # x = tf.cast(x, tf.float64)
    # y = tf.cast(y, tf.int64)
    return (x, y)

def get_train_data(batch_size):
    filenames = ['./data_tens/traindata-00.tfrecord', './data_tens/traindata-01.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000, num_parallel_reads=2)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset

def get_valid_data(batch_size):
    filenames = ['./data_tens/validdata-00.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def get_test_data(batch_size):
    filenames = ['./data_tens/testdata-00.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


# print(get_train_data(100))
# data=get_train_data(100)
# for batch in data.take(1):
#     x_batch, y_batch = batch
#     # print("Batch y labels:")
#     print(x_batch)

