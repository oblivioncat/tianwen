import tensorflow as tf
from batch_data_reader import BatchDataset
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from scipy import signal, ndimage
from astropy.convolution import Gaussian1DKernel,convolve

train_txt_dir = '../second_b_train_data_20180313'
train_idx_file = './second_b_train_index_20180313.csv'
test_txt_dir = '../second_b_test_data_20180313'
test_idx_file = './second_b_test_index_20180313.csv'

# train_txt_dir_1 = '../first_train_data_20180131'
# train_idx_file_1 = './first_train_index_20180131.csv'

weights = [1, 1, 1, 1]
n_per_class = 50000

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _parse(file):
    with open(train_txt_dir + os.sep + str(file) + '.txt', 'r') as f:
        content = f.readline().rstrip('\n').split(',')

    content = map(np.float32, content)
    # content = preprocessing(content)

    return np.array(content)

def _parse_test(file, file_path):
    with open(file_path + os.sep + str(file) + '.txt', 'r') as f:
        content = f.readline().rstrip('\n').split(',')

    content = map(np.float32, content)
    # content = preprocessing(content)

    return np.array(content)

def convert_to(data_set, name):
    filename = os.path.join(name + '.tfrecords')

    with tf.python_io.TFRecordWriter(filename) as writer:
        for idx in tqdm(range(data_set.shape[0])):
            x = _parse(data_set[idx, 0])
            y = data_set[idx, 1]

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                    'feature': _bytes_feature(x.tostring()),
                    'label': _int64_feature(int(y))
                    }))

            writer.write(example.SerializeToString())

def convert_to_test(data_set, name, file_path):
    filename = os.path.join(name + '.tfrecords')

    with tf.python_io.TFRecordWriter(filename) as writer:
        for idx in tqdm(range(data_set.shape[0])):
            x = _parse_test(data_set[idx],file_path)
            y = -1
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'feature': _bytes_feature(x.tostring()),
                        'label': _int64_feature(int(y))
                    }))

            writer.write(example.SerializeToString())

data_sets = BatchDataset(valid_size=0.2)
data_sets.refresh_rnd_train(weights=weights, npc=n_per_class)

print('Writing train.tfrecords')
convert_to(data_sets.rnd_train, 'train_b_dm')

print('Writing validation.tfrecords')
convert_to(data_sets.rnd_valid, 'validation_b_dm')

print('Writing test.tfrecords')
# df = pd.read_csv(test_idx_file).as_matrix()
df=pd.read_csv(test_idx_file).as_matrix()
rnd_test = np.squeeze(df)
convert_to_test(rnd_test, 'test_b_dm', test_txt_dir)