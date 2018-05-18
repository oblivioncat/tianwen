import tensorflow as tf
import time
import os
from sklearn.preprocessing import scale
from utils import preprocessing
# from batch_data_reader import BatchDataset
from datetime import datetime
import argparse
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
Inception-resnet
"""

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'

exp = 3

model_path = './model'
log_path = './log'
result_path = './result_incepres%d/' % exp


test_idx_file = './second_a_test_index_20180313.csv'

train_size = 353676
valid_size = 86769
test_size = 100000

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=int, default=[1,1,1,1])
parser.add_argument('--channels', nargs='+', type=int, default=[16,16,16,16])
parser.add_argument('--ks', nargs='+', type=int, default=[40,40,20,20])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--reg', type=float, default=0.0)
# parser.add_argument('--round', type=int, default=1)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--npc', type=int, default=50000)
parser.add_argument('--batch', type=int, default=400)
# parser.add_argument('--oversampling', type=str, default=None)
flags = parser.parse_args()

n_classes = 4

def augment(feature, label):
    def _func(feature, label):
        feature = preprocessing(feature, None)
        feature = scale(feature, axis=-1)

        return feature.astype(np.float32), label

    return tf.py_func(_func, [feature, label], [tf.float32, tf.int64])

def decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'feature': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    feature = tf.decode_raw(features['feature'], tf.float32)
    feature.set_shape([2600])

    label = tf.cast(features['label'], tf.int64)

    return feature, label

def inputs(handle, batch_size):
    def parse_record(file_name, train=True):
        dataset = tf.data.TFRecordDataset(file_name)
        dataset = dataset.repeat()
        dataset = dataset.map(decode)
        dataset = dataset.map(augment)

        if train:
            dataset = dataset.shuffle(3 * batch_size)
        dataset = dataset.batch(batch_size)

        return dataset

    train_dataset = parse_record(TRAIN_FILE, train=True)
    valid_dataset = parse_record(VALIDATION_FILE, train=False)
    test_dataset = parse_record(TEST_FILE, train=False)

    iterator = tf.data.Iterator.from_string_handle(handle,
                                                   train_dataset.output_types,
                                                   train_dataset.output_shapes)

    train_iterator = train_dataset.make_one_shot_iterator()
    valid_iterator = valid_dataset.make_one_shot_iterator()
    test_iterator = test_dataset.make_one_shot_iterator()

    return train_iterator, valid_iterator, test_iterator, iterator.get_next()

def stem(input):
    """

    :param input: ?,2600
    :return: ?,568,392
    """

    conv1 = tf.layers.conv1d(inputs=input, filters=64, kernel_size=41, strides=2,
                             padding='valid', activation=None, name='conv_%d' % 1)
    relu1 = tf.nn.relu(conv1, name='relu_%d' % 1)
    # relu1 (?,1280,64)
    conv2 = tf.layers.conv1d(inputs=relu1, filters=64, kernel_size=41, strides=1,
                             padding='valid', activation=None, name='conv_%d' % 2)
    relu2 = tf.nn.relu(conv2, name='relu_%d' % 2)
    # relu2 ?,1240,64
    conv3 = tf.layers.conv1d(inputs=relu2, filters=128, kernel_size=41, strides=1,
                             padding='same', activation=None, name='conv_%d' % 3)
    relu3 = tf.nn.relu(conv3, name='relu_%d' % 2)
    # relu3 ?,1240,128
    # conv4_1 = tf.layers.conv1d(inputs=relu1, filters=64, kernel_size=41, strides=2,
    #                            padding='valid', activation=tf.nn.relu, name='conv_%d' % 4)
    # conv4_1 ?,580,96
    pool4_2 = tf.layers.max_pooling1d(inputs=relu3, pool_size=2, strides=2,
                                      padding='valid', name='pool_%d' % 4)
    # pool4_2 ?,620,128
    # concat1 = tf.concat([conv4_1, pool4_2], 2)
    # concat1 ?,580,128
    conv5_11 = tf.layers.conv1d(inputs=pool4_2, filters=64, kernel_size=1, strides=1,
                                padding='same', activation=tf.nn.relu, name='conv_%d' % 511)

    conv5_12 = tf.layers.conv1d(inputs=conv5_11, filters=64, kernel_size=10, strides=1,
                                padding='same', activation=tf.nn.relu, name='conv_%d' % 512)

    conv5_13 = tf.layers.conv1d(inputs=conv5_12, filters=96, kernel_size=11, strides=1,
                                padding='valid', activation=tf.nn.relu, name='conv_%d' % 513)
    # # conv5_13 ?,610,96
    # conv5_21 = tf.layers.conv1d(inputs=concat1, filters=64, kernel_size=1, strides=1,
    #                             padding='same', activation=None, name='conv_%d' % 521)
    #
    # conv5_22 = tf.layers.conv1d(inputs=conv5_21, filters=96, kernel_size=11, strides=1,
    #                               padding='valid', activation=None, name='conv_%d' % 522)
    # # conv5_22 ?,594,96
    # concat2 = tf.concat([conv5_13, conv5_22], 2)
    # concat2 ?,574,192
    # conv6_1 = tf.layers.conv1d(inputs=conv4_1, filters=128, kernel_size=21, strides=1,
    #                            padding='valid', activation=tf.nn.relu, name='conv_%d' % 6)
    # pool6_2 = tf.layers.max_pooling1d(inputs=conv4_1, pool_size=21, strides=1,
    #                                   padding='valid', name='pool_%d' % 6)
    # pool6_2 ?,622,192
    # concat3 = tf.concat([conv6_1, pool6_2], 2)
    # concat3 ?,622,384
    return conv5_13

def Incep_res_A(input, alpha=1.0):
    input = tf.nn.relu(input,name = 'A_input')
    shape = input.get_shape().as_list()
    # input ?,568,392
    conv_1 = tf.layers.conv1d(inputs=input, filters=16, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, name='A_conv_%d' % 1)
    # conv_a ?,568,32
    conv_21 = tf.layers.conv1d(inputs=input, filters=16, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, name='A_conv_%d' % 21)

    conv_22 = tf.layers.conv1d(inputs=conv_21, filters=16, kernel_size=21, strides=1,
                               padding='same', activation=tf.nn.relu, name='A_conv_%d' % 22)
    # conv_22 ?,568,32
    conv_31 = tf.layers.conv1d(inputs=input, filters=16, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, name='A_conv_%d' % 31)

    conv_32 = tf.layers.conv1d(inputs=conv_31, filters=16, kernel_size=41, strides=1,
                               padding='same', activation=tf.nn.relu, name='A_conv_%d' % 32)

    # conv_33 = tf.layers.conv1d(inputs=conv_32, filters=32, kernel_size=41, strides=1,
    #                            padding='same', activation=tf.nn.relu, name='A_conv_%d' % 33)

    conv_41 = tf.layers.conv1d(inputs=input, filters=16, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, name='A_conv_%d' % 41)

    conv_42 = tf.layers.conv1d(inputs=conv_41, filters=16, kernel_size=81, strides=1,
                               padding='same', activation=tf.nn.relu, name='A_conv_%d' % 42)

    # conv_33 ?,568,64
    concat1 = tf.concat([conv_1, conv_22, conv_32, conv_42],2)
    # concat1 ?,622,128
    conv_4 = tf.layers.conv1d(inputs=concat1, filters=shape[2], kernel_size=1, strides=1,
                               padding='same', activation=None, name='A_conv_%d' % 4)
    # conv_4 ?,622,384
    output = tf.add((tf.multiply(input,alpha)),conv_4)
    # concat2 ?,622,384
    relu = tf.nn.relu(output, name='A_relu')

    return relu

def Incep_res_B(input,alpha=1.0):
    input = tf.nn.relu(input, name='B_input')
    shape = input.get_shape().as_list()
    # input 280,1152
    conv_1 = tf.layers.conv1d(inputs=input, filters=64, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, name='B_conv_%d' % 1)
    # conv1 ?,280,192
    conv_21 = tf.layers.conv1d(inputs=input, filters=64, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, name='B_conv_%d' % 21)

    conv_22 = tf.layers.conv1d(inputs=conv_21, filters=64, kernel_size=3, strides=1,
                               padding='same', activation=tf.nn.relu, name='B_conv_%d' % 22)

    conv_31 = tf.layers.conv1d(inputs=input, filters=64, kernel_size=1, strides=1,
                               padding='same', activation=tf.nn.relu, name='B_conv_%d' % 31)

    conv_32 = tf.layers.conv1d(inputs=conv_31, filters=64, kernel_size=7, strides=1,
                               padding='same', activation=tf.nn.relu, name='B_conv_%d' % 32)
    # conv_22 280,192
    concat1 = tf.concat([conv_1, conv_22, conv_32], 2)
    # concat1 309,192
    conv_4 = tf.layers.conv1d(inputs=concat1, filters=shape[2], kernel_size=1, strides=1,
                               padding='same', activation=None, name='B_conv_%d' % 4)
    # conv3 309,128
    output = tf.add((tf.multiply(input,alpha)), conv_4)
    # output 309,128
    relu = tf.nn.relu(output, name='B_relu')

    return relu

def Incep_res_C(input, alpha=1.0):
    input = tf.nn.relu(input, name = 'C_iput')
    shape = input.get_shape().as_list()

    conv_1 = tf.layers.conv1d(inputs=input, filters=192, kernel_size=1, strides=1,
                              padding='same', activation=None, name='C_conv_%d' % 1)

    conv_21 = tf.layers.conv1d(inputs=input, filters=192, kernel_size=1, strides=1,
                               padding='same', activation=None, name='C_conv_%d' % 21)

    conv_22 = tf.layers.conv1d(inputs=conv_21, filters=256, kernel_size=10, strides=1,
                               padding='same', activation=None, name='C_conv_%d' % 22)

    concat1 = tf.concat([conv_1, conv_22], 2)

    conv_3 = tf.layers.conv1d(inputs=concat1, filters=shape[2], kernel_size=1, strides=1,
                              padding='same', activation=None, name='C_conv_%d' % 3)

    output = tf.add((tf.multiply(input, alpha)), conv_3)
    # output 136,2048
    relu = tf.nn.relu(output, name='C_relu')

    return relu

def Reduction_A(input):

    pool1 = tf.layers.max_pooling1d(inputs=input, pool_size=3, strides=2,
                                      padding='valid', name='AR_pool')
    # pool1 280,384
    conv1 = tf.layers.conv1d(inputs=input, filters=32, kernel_size=3, strides=2,
                              padding='valid', activation=tf.nn.relu, name='RA_conv_%d' % 1)
    # conv1 280,384
    conv21 = tf.layers.conv1d(inputs=input, filters=32, kernel_size=1, strides=1,
                             padding='same', activation=tf.nn.relu, name='RA_conv_%d' % 21)

    # conv22 = tf.layers.conv1d(inputs=conv21, filters=32, kernel_size=3, strides=1,
    #                          padding='same', activation=tf.nn.relu, name='RA_conv_%d' % 22)

    conv23 = tf.layers.conv1d(inputs=conv21, filters=32, kernel_size=3, strides=2,
                             padding='valid', activation=tf.nn.relu, name='RA_conv_%d' % 23)
    # conv23 285,384
    concat = tf.concat([pool1, conv1, conv23], 2)
    # concat 309,128
    return concat

def Reduction_B(input):
    # input 280,1152
    pool1 = tf.layers.max_pooling1d(inputs=input, pool_size=10, strides=2,
                                    padding='valid', name='RB_pool')
    # pool1 136,2432
    conv11 = tf.layers.conv1d(inputs=input, filters=256, kernel_size=1, strides=1,
                             padding='same', activation=tf.nn.relu, name='RB_conv_%d' % 11)
    # conv11 280,256
    conv12 = tf.layers.conv1d(inputs=conv11, filters=384, kernel_size=10, strides=2,
                              padding='valid', activation=tf.nn.relu, name='RB_conv_%d' % 12)
    # conv12 136,384
    conv21 = tf.layers.conv1d(inputs=input, filters=256, kernel_size=1, strides=1,
                             padding='same', activation=tf.nn.relu, name='RB_conv_%d' % 21)

    conv22 = tf.layers.conv1d(inputs=conv21, filters=256, kernel_size=10, strides=2,
                              padding='valid', activation=tf.nn.relu, name='RB_conv_%d' % 22)
    # conv22 136,256
    conv31 = tf.layers.conv1d(inputs=input, filters=256, kernel_size=1, strides=1,
                              padding='same', activation=tf.nn.relu, name='RB_conv_%d' % 31)

    conv32 = tf.layers.conv1d(inputs=conv31, filters=256, kernel_size=10, strides=1,
                              padding='same', activation=tf.nn.relu, name='RB_conv_%d' % 32)

    conv33 = tf.layers.conv1d(inputs=conv32, filters=256, kernel_size=10, strides=2,
                              padding='valid', activation=tf.nn.relu, name='RB_conv_%d' % 33)
    # conv33 136,256
    concat = tf.concat([pool1, conv12, conv22, conv33], 2)
    # concat 136,2048
    return concat



def make_net(input, keep_prob, channels, kernel_size, training=True, reg=0.01):
    input = tf.expand_dims(input, axis=-1)
    feat_len = input.get_shape().as_list()[1]

    stem_output = stem(input)
    # stem_output ?,620,64
    Incep_A = Incep_res_A(stem_output)
    # Incep_A ?,620,64
    Reduc_A = Reduction_A(Incep_A)
    # Reduc_A ? 309,128
    Incep_B = Incep_res_B(Reduc_A, alpha=1)
    # Incep_B 309,128
    # Reduc_B = Reduction_B(Incep_B)
    # Reduc_B 136,2048
    # Incep_C = Incep_res_C(Reduc_B)
    # Incep_C 136,2048
    shape_B = Incep_B.get_shape().as_list()
    Ave_pool = tf.layers.average_pooling1d(Incep_B, pool_size=shape_B[1], strides=1)
    # Ave_pool 310,896
    FC = tf.reshape(Ave_pool,[-1, shape_B[2]])
    FC = tf.nn.dropout(FC, keep_prob=keep_prob)
    # output = tf.layers.dense(FC,448,activation=tf.nn.relu)
    # output = tf.nn.dropout(output, keep_prob=keep_prob)
    output = tf.layers.dense(FC, n_classes, activation=tf.nn.relu)

    return output

def main():
    weights = flags.weights
    channels = flags.channels
    kernel_size = flags.ks
    learning_rate = flags.lr
    l2_reg = flags.reg
    # n_round = flags.round
    n_epochs = flags.epoch
    # n_per_class = flags.npc
    batch_size = flags.batch
    # wl = flags.wavelet
    # oversampling = flags.oversampling

    # n = 0
    # for w in weights:
    #     n += w
    # step_per_epoch = n * n_per_class / batch_size
    step_per_epoch = train_size / batch_size

    channel_str = ''
    for channel in channels:
        channel_str += '-' + str(channel)
    channel_str = channel_str[1:]
    weight_str = ''
    for weight in weights:
        weight_str += '-' + str(weight)
    weight_str = weight_str[1:]
    kernel_str = ''
    for kernel in kernel_size:
        kernel_str += '-' + str(kernel)
    kernel_str = kernel_str[1:]

    stamp = ('incepres%d' % (exp))

    log_dir = log_path + os.sep + stamp
    model_dir = model_path + os.sep + stamp

    tf.reset_default_graph()
    tf.set_random_seed(1234)

    handle = tf.placeholder(tf.string, shape=[])
    mode = tf.placeholder(tf.bool, name='mode')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    train_iterator, valid_iterator, test_iterator, (x, y) = inputs(handle, batch_size)

    # After call py_func, (x, y) shape info lost
    x.set_shape([None, 2600])
    y.set_shape([None])


    logits = make_net(x, keep_prob=0.4, channels=channels, kernel_size=kernel_size, training=mode, reg=l2_reg)
    tf.summary.histogram('logits', logits)

    batch_weights=tf.gather(weights,y)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits,weights=batch_weights)
    reg_loss = tf.losses.get_regularization_loss()
    loss += reg_loss
    tf.summary.scalar('loss', loss)

    global_step = tf.train.get_or_create_global_step()
    start_learning_rate = learning_rate
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                               step_per_epoch, 0.96, staircase=True)
    tf.summary.scalar('lr', learning_rate)
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optim.minimize(loss, global_step)

    y_pred = tf.argmax(logits, axis=1)
    # Compute accuracy per class
    acc_list = []
    for cls in range(n_classes):
        y_mask = tf.equal(y, cls)
        y_pred_mask = tf.equal(y_pred, cls)
        acc_per_cls = (tf.reduce_sum(tf.cast(tf.logical_and(y_mask, y_pred_mask), tf.float32)) /
                       (tf.reduce_sum(tf.cast(y_pred_mask, tf.float32)) + 1e-7))
        acc_list.append(acc_per_cls)
        tf.summary.scalar('acc_%d' % cls, acc_per_cls)

    # Compute recall per class
    recall_list = []
    for cls in range(n_classes):
        y_mask = tf.equal(y, cls)
        y_pred_mask = tf.equal(y_pred, cls)
        recall_per_cls = (tf.reduce_sum(tf.cast(tf.logical_and(y_mask, y_pred_mask), tf.float32)) /
                          (tf.reduce_sum(tf.cast(y_mask, tf.float32)) + 1e-7))
        recall_list.append(recall_per_cls)
        tf.summary.scalar('recall_%d' % cls, recall_per_cls)

    # Macro f1 score
    macro_f1 = 0.0
    for i in range(n_classes):
        f1_per_cls = 2 * acc_list[i] * recall_list[i] / (acc_list[i] + recall_list[i] + 1e-7)
        tf.summary.scalar('f1_score_%d' % i, f1_per_cls)
        macro_f1 += 2 * acc_list[i] * recall_list[i] / (acc_list[i] + recall_list[i] + 1e-7) / n_classes
    tf.summary.scalar('macro_f1', macro_f1)

    # Compute proportion per class in prediction
    for cls in range(n_classes):
        ratio = tf.reduce_mean(tf.cast(tf.equal(y_pred, cls), tf.float32))
        tf.summary.scalar('y_pred_%d_ratio' % cls, ratio)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
    tf.summary.scalar('acc', accuracy)

    tf.add_to_collection("inputs", x)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("logits",logits)
    tf.add_to_collection("y_pred",y_pred)
    tf.add_to_collection("outputs", y_pred)

    summary_op = tf.summary.merge_all()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Remove last training record
        shutil.rmtree(log_dir, ignore_errors=True)
        shutil.rmtree(log_dir + '_val', ignore_errors=True)

        train_summary_writer = tf.summary.FileWriter(log_dir)
        valid_summary_writer = tf.summary.FileWriter(log_dir + '_val')
        init = tf.global_variables_initializer()
        sess.run(init)


        saver = tf.train.Saver(max_to_keep=n_epochs)

        train_handler = sess.run(train_iterator.string_handle())
        valid_handler = sess.run(valid_iterator.string_handle())
        test_handler = sess.run(test_iterator.string_handle())


        for step in xrange(n_epochs * step_per_epoch):
            start_time = time.time()
            _, loss_val, step_summary, global_step_val = sess.run(
                [train_op, loss, summary_op, global_step],
                feed_dict={mode: True, keep_prob: 0.4, handle: train_handler}
            )
            duration = time.time() - start_time
            format_str = ('%s: step %d, loss = %.2f (%.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), global_step_val, loss_val, duration))
            # print('batch weights', weights)

            train_summary_writer.add_summary(step_summary, global_step_val)

            if step % 5 == 0:
                step_summary, global_step_val = sess.run(
                    [summary_op, global_step], feed_dict={mode: False,
                                                          keep_prob: 1.0,
                                                          handle: valid_handler}
                )
                valid_summary_writer.add_summary(step_summary, global_step_val)

            if step % step_per_epoch == 0 and (step / step_per_epoch) > 0:
                # Save model
                save_path = "%s/ep%d" % (model_dir, step / step_per_epoch)
                shutil.rmtree(save_path, ignore_errors=True)
                os.makedirs(save_path)
                saver.save(sess, "%s/model.ckpt" % save_path)

                test_pred = []
                test_logit = []
                print('Apply model on valid set')
                for _ in tqdm(range(test_size / batch_size)):
                    logits_val, y_pred_val = sess.run([tf.nn.softmax(logits), y_pred],
                                                      feed_dict={mode: False,
                                                                 keep_prob: 1.0,
                                                                 handle: test_handler})
                    test_pred.extend(y_pred_val)
                    test_logit.extend(logits_val)

                label_map = {'qso': 3, 'star': 0, 'unknown': 1, 'galaxy': 2}
                # Exchange key and value
                label_map = dict((k, v) for v, k in label_map.items())

                test_pred_map = [label_map[pp] for pp in test_pred]
                test_pred_map = np.array(test_pred_map)
                test_pred_map = np.expand_dims(test_pred_map, axis=-1)
                test_logit = np.array(test_logit)

                if not os.path.exists(result_path):
                    os.mkdir(result_path)

                test_pred_file = result_path+'%s_ep%d.csv' % (stamp, step // step_per_epoch)
                test_logit_file = result_path+'%s_ep%d_lgt.csv' % (stamp, step // step_per_epoch)
                test_stat_file = result_path+'%s_ep%d_stat.csv' % (stamp, step // step_per_epoch)


                test_idx = pd.read_csv(test_idx_file).as_matrix()
                merge_pred = np.concatenate((test_idx, test_pred_map), axis=-1)
                df = pd.DataFrame(merge_pred)
                df.to_csv(test_pred_file, header=False, index=False)

                series = df.iloc[:, -1]
                stat = series.value_counts()
                stat.to_csv(test_stat_file, header=False, index=False)

                merge_logit = np.concatenate((test_idx, test_logit), axis=-1)
                df = pd.DataFrame(merge_logit)
                df.to_csv(test_logit_file, header=['id', 'star', 'unknown', 'galaxy', 'qso'], index=False)

        # coord.request_stop()
        # coord.join(threads)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()