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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

TRAIN_FILE = 'train_b.tfrecords'
VALIDATION_FILE = 'validation_b.tfrecords'
TEST_FILE = 'test_b.tfrecords'

exp = 2
model_path = './model'
log_path = './log'
result_path = './result_incep%d/' % exp


test_idx_file = './second_b_test_index_20180313.csv'

train_size = 347083
valid_size = 86768
test_size = 100000

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=int, default=[1,1,1,1])
parser.add_argument('--channels', nargs='+', type=int, default=[16,16,32,32])
parser.add_argument('--ks', nargs='+', type=int, default=[40,40,20,20])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--reg', type=float, default=0.0)
# parser.add_argument('--round', type=int, default=1)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--npc', type=int, default=50000)
parser.add_argument('--batch', type=int, default=1000)
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

def inception(input,channels,index):
    # conv 1x1, input, channels/2
    output1 = tf.layers.conv1d(inputs=input, filters=2*channels, kernel_size=1, strides=1,
                               padding='same', activation=None,name = 'conv_%s%d' % (index,11))

    # conv 1x1, input, channel/2 and conv 3x3, input, channel/2
    # output2 = tf.layers.conv1d(inputs=input, filters=channels, kernel_size=1, strides=1,
    #                            padding='same', activation=None, name = 'conv_%s%d' % (index,21))

    output2 = tf.layers.conv1d(inputs=input, filters=2*channels, kernel_size=10, strides=1,
                               padding='same', activation=None, name='conv_%s%d' % (index, 22))

    # conv 5x5, input, channel/2
    # output3 = tf.layers.conv1d(inputs=input, filters=channels, kernel_size=1, strides=1,
    #                            padding='same', activation=None, name = 'conv_%s%d' % (index,31))

    output3 = tf.layers.conv1d(inputs=input, filters=channels, kernel_size=20, strides=1,
                               padding='same', activation=None, name='conv_%s%d' % (index, 32))

    # max pooling
    output4 = tf.layers.max_pooling1d(inputs=input, strides=1,pool_size=10,
                               padding='same', name = 'pool_%s%d' % (index,41))

    # output4 = tf.layers.conv1d(inputs=output4, filters=channels, kernel_size=3, strides=1,
    #                            padding='same', activation=None, name='conv_%s%d' % (index, 42))

    output = tf.concat([output1, output2, output3, output4], 2)

    return output

def inception_red(input, channels, kernel_size, index,res=False):
    # conv 1x1, input, channels/2
    output1 = tf.layers.conv1d(inputs=input, filters= channels/2, kernel_size=1, strides=1,
                               padding='same', activation=None, name='conv_%s%d' % (index, 11))

    # conv 1x1, input, channel/2 and conv 3x3, input, channel/2
    output2 = tf.layers.conv1d(inputs=input, filters=channels/2, kernel_size=1, strides=1,
                               padding='same', activation=None, name = 'conv_%s%d' % (index,21))

    output2 = tf.layers.conv1d(inputs=output2, filters=channels/2, kernel_size=kernel_size[0], strides=1,
                               padding='same', activation=None, name='conv_%s%d' % (index, 22))

    # conv 5x5, input, channel/2
    output3 = tf.layers.conv1d(inputs=input, filters=channels/2, kernel_size=1, strides=1,
                               padding='same', activation=None, name = 'conv_%s%d' % (index,31))

    output3 = tf.layers.conv1d(inputs=output3, filters=channels/2, kernel_size=kernel_size[1], strides=1,
                               padding='same', activation=None, name='conv_%s%d' % (index, 32))

    # 3x3 max pooling
    output4 = tf.layers.max_pooling1d(inputs=input, strides=1, pool_size=kernel_size[2],
                                      padding='same', name='pool_%s%d' % (index, 41))

    output4 = tf.layers.conv1d(inputs=output4, filters=channels/2, kernel_size=1, strides=1,
                               padding='same', activation=None, name='conv_%s%d' % (index, 42))

    if res == True:
        output = tf.concat([output1, output2, output3, output4, input], 2)
    else:
        output = tf.concat([output1, output2, output3, output4], 2)

    return output

def make_net(input, keep_prob, channels, kernel_size, training=True, reg=0.01):
    input = tf.expand_dims(input, axis=-1)
    feat_len = input.get_shape().as_list()[1]

    conv1 = tf.layers.conv1d(inputs=input, filters=32, kernel_size=40, strides=2,
                             padding='same', activation=None, name='conv_%d' % 1)
    relu1 = tf.nn.relu(conv1, name='relu_%d' % 1)
    pool1 = tf.layers.max_pooling1d(inputs=relu1, pool_size=2, strides=2,
                                    padding='same', name='pool_%d' % 1)

    conv2 = tf.layers.conv1d(inputs=pool1, filters=32, kernel_size=20, strides=1,
                             padding='same', activation=None, name='conv_%d' % 2)
    relu2 = tf.nn.relu(conv2, name='relu_%d' % 2)
    pool2 = tf.layers.max_pooling1d(inputs=relu2, pool_size=2, strides=2,
                                    padding='same', name='pool_%d' % 2)

    incep1 = inception_red(pool2, channels=32, kernel_size=[10,20,5], index='incep_a')
    # incep1: 325, 2 * channels
    pool3 = tf.layers.max_pooling1d(inputs=incep1, pool_size=2, strides=2,
                                    padding='same', name='pool_%d' % 3)

    incep2 = inception_red(pool3, channels=32, kernel_size=[3,7,3], index='incep_b')
    # incep2: 163, 2 * channels
    incep3 = inception_red(incep2, channels=32,kernel_size=[3,5,3], index='incep_c')
    # incep3: 163, 2 * channels
    pool4 = tf.layers.max_pooling1d(inputs=incep3, pool_size=2, strides=2,
                                    padding='same', name='pool_%d' % 4)

    incep4 = inception_red(pool4, channels=64, kernel_size=[3,5,3], index='incep_d')
    # incep4 82, 2 * channels
    # incep3 = inception(pool1, channels=8,index='incep_c')

    import math
    final_size = int(math.ceil(feat_len / 2.0 ** 5))
    AP = tf.layers.average_pooling1d(incep4,final_size,strides=1)
    FC = tf.nn.dropout(AP, keep_prob=keep_prob)
    FC = tf.reshape(FC,[-1, 128])
    return tf.layers.dense(FC, n_classes, activation=None)


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

    stamp = ('incep%d' % (exp))

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

        # r = 0
        # saver = tf.train.Saver()
        # if os.path.exists(flags.ckdir) and tf.train.checkpoint_exists(flags.ckdir):
        #     r = r+1
        #     latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
        #     saver.restore(sess, latest_check_point)

        # global_step = tf.train.get_global_step(sess.graph)

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)

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

    main()