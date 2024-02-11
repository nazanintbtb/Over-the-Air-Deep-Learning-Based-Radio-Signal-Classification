import h5py
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
def serialize_example(x, y):
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    feature = {
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized_example = example_proto.SerializeToString()
    return serialized_example

def traindata_to_tfrecord(x,y):

    for file_num in range(2):
        with tf.io.TFRecordWriter('./data_tens/traindata-%.2d.tfrecord' % file_num) as writer:
            for i in tqdm(range(file_num * int(len(y)/2), (file_num + 1) * int(len(y)/2)), desc="Processing Train Data {}".format(file_num), ascii=True):
                example_proto = serialize_example(x[i], y[i])
                writer.write(example_proto)


def testdata_to_tfrecord(x, y):
    for file_num in range(1):
        with tf.io.TFRecordWriter('./data_tens/testdata-%.2d.tfrecord' % file_num) as writer:
            for i in tqdm(range(file_num * len(y), (file_num + 1) * len(y)), desc="Processing Test Data {}".format(file_num), ascii=True):
                example_proto = serialize_example(x[i], y[i])
                writer.write(example_proto)

def validdata_to_tfrecord(x, y):
    for file_num in range(1):
        with tf.io.TFRecordWriter('./data_tens/validdata-%.2d.tfrecord' % file_num) as writer:
            for i in tqdm(range(file_num * len(y), (file_num + 1) * len(y)), desc="Processing valid Data {}".format(file_num), ascii=True):
                example_proto = serialize_example(x[i], y[i])
                writer.write(example_proto)


if __name__ == "__main__":

    filename = './dataset.h5'
    with h5py.File(filename, 'r') as file:
        x = file['signals'][:]
        y = file['labels'][:]

        # print("Original Shapes:")
        # print("x shape:", x.shape)
        # print("y shape:", y.shape)

        indices = np.arange(len(y))
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        x_train, x_temp, y_train, y_temp = train_test_split(x_shuffled, y_shuffled, test_size=0.15, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.33, random_state=42)

        # print("\nShapes after Split:")
        # print("Train:", x_train.shape, y_train.shape)
        # print("Validation:", x_val.shape, y_val.shape)
        # print("Test:", x_test.shape, y_test.shape)
        traindata_to_tfrecord(x_train,y_train)
        testdata_to_tfrecord(x_test,y_test)
        validdata_to_tfrecord(x_val,y_val)