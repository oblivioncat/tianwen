import pandas as pd
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
from sklearn.preprocessing import scale
import os
# from imblearn.over_sampling import RandomOverSampler
from utils import preprocessing
from scipy import signal, ndimage
from astropy.convolution import Gaussian1DKernel,convolve
import os, shutil

train_txt_dir = '../second_b_train_data_20180313'
train_idx_file = './second_b_train_index_20180313.csv'
train_txt_dir_1 = '../first_train_data_20180131'
train_idx_file_1 = './first_train_index_20180131.csv'
n_classes = 4

class BatchDataset:
    raw_train = {key: [] for key in range(n_classes)}
    raw_valid = {key: [] for key in range(n_classes)}
    rnd_train = []
    rnd_valid = []
    used = {key: [] for key in range(n_classes)}
    batch_offset = 0
    batch_offset_val = 0

    def __init__(self, valid_size=0.2):
        print('initialize:')
        label_map = {'qso': 3, 'unknown': 1, 'star': 0, 'galaxy': 2}
        # label_map = {'qso': 2, 'star': 0, 'galaxy': 1}

        train_csv = pd.read_csv(train_idx_file)

        # train_index = pd.read_csv('./first_train_index_20180131.csv')
        # train_qso = train_index[train_index['type'].isin(['qso'])]
        # train_galaxy = train_index[train_index['type'].isin(['galaxy'])]
        #
        # train_temp=pd.merge(train_qso, train_galaxy, 'outer')
        # train_1 = pd.merge(train_temp,train_csv,'outer')

        id, label = train_csv['id'], train_csv['type']

        id = list(id); label = list(label)

        # data = {0: [], 1: [], 2: [], 3: [],}
        data = {key: [] for key in range(n_classes)}
        counter = {key: 0 for key in range(n_classes + 1)}
        for i, l in zip(id, label):
            data[label_map[l]].append(i)
            counter[label_map[l]] += 1
            counter[n_classes] += 1
        print(counter)

        # valid_len = int(len(data[3]) * valid_size)
        for k in data:
            # Imbalance valid dataset
            valid_len = int(len(data[k]) * valid_size)
            l = len(data[k])
            valid_idx = random.sample(range(l), valid_len)
            train_idx = set(range(l)) - set(valid_idx)
            train_idx = list(train_idx)

            self.raw_train[k] = [data[k][idx] for idx in train_idx]
            self.raw_valid[k] = [data[k][idx] for idx in valid_idx]

        # extra training data
        # train_index = pd.read_csv('./first_train_index_20180131.csv')
        # train_qso = train_index[train_index['type'].isin(['qso'])]
        # train_galaxy = train_index[train_index['type'].isin(['galaxy'])]
        # train_unkn = train_index[train_index['type'].isin(['unknown'])]
        # train_star = train_index[train_index['type'].isin(['star'])]
        # l_1 = len(train_star)
        # l_2 = len(train_unkn)
        # l_3 = len(train_galaxy)
        # l_4 = len(train_qso)
        #
        # star_rnd = random.sample(range(l_1),l_3+l_4)
        # unkn_rnd = random.sample(range(l_2),l_3+l_4)
        # star_name = [train_star['id'].values[idx] for idx in star_rnd]
        #
        # for i in star_name:
        #     shutil.copyfile('../first_train_data_20180131/' + str(i) + '.txt', '../second_a_train_data_20180313/' + str(i) + '.txt')
        #
        # unkn_name = [train_unkn['id'].values[idx] for idx in unkn_rnd]
        # for i in unkn_name:
        #     shutil.copyfile('../first_train_data_20180131/' + str(i) + '.txt', '../second_a_train_data_20180313/' + str(i) + '.txt')
        #
        # list_tosave = star_name + unkn_name
        # df_tosave = pd.DataFrame(list_tosave)
        # df_tosave.to_csv('./rnd_star_galaxy.csv', index=False)

        # self.raw_train[0] = self.raw_train[0] + star_name
        # self.raw_train[1] = self.raw_train[1] + unkn_name
        # self.raw_train[2] = self.raw_train[2] + list(train_galaxy['id'].values)
        # self.raw_train[3] = self.raw_train[3] + list(train_qso['id'].values)

        print('The len of train_data is :',len(self.raw_train[0])+len(self.raw_train[1])+len(self.raw_train[2])+len(self.raw_train[3]))

        # Init rnd valid
        for k in self.raw_valid:
            valid_len = int(len(data[k]) * valid_size)
            self.rnd_valid.extend([[self.raw_valid[k][i], k] for i in range(valid_len)])

        # Shuffle rnd valid
        self.rnd_valid = np.array(self.rnd_valid)
        perm = np.arange(self.rnd_valid.shape[0])
        np.random.shuffle(perm)
        self.rnd_valid = self.rnd_valid[perm]

    def _parse(self, file):
        with open(train_txt_dir + os.sep + str(file) + '.txt', 'r') as f:
            content = f.readline().rstrip('\n').split(',')

        content = map(np.float32, content)
        content = preprocessing(content)

        return content

    def refresh_rnd_train(self, weights=None, npc=50000, reuse=None):
        print('Refresh rnd train')
        self.rnd_train = []

        # npc_copy = npc
        for k in self.raw_train:
            # prior = []
            # npc = npc_copy * weights[k]
            # if reuse is not None and len(reuse[k]) > 0:
            #     prior = np.random.choice(reuse[k], size=int(npc * 0.25))
            #     npc = int(npc * 0.75)

            l = len(self.raw_train[k])
            # if npc > l:
            #     s = range(l)
            #     padding = np.random.randint(l, size=npc - l)
            #     s.extend(padding)
            # else:
            #     c = set(range(l)) - set(self.used[k])
            #     c = list(c)
            #     if npc > len(c):
            #         s = random.sample(range(l), npc)
            #         self.used[k] = []
            #     else:
            #         s = random.sample(c, npc)
            # self.used[k].extend(s)
            # s.extend(prior)
            # self.rnd_train.extend([[self.raw_train[k][js], k, js] for js in s])
            self.rnd_train.extend([[self.raw_train[k][js], k] for js in range(l)])

        # Shuffle rnd train
        self.rnd_train = np.array(self.rnd_train)
        perm = np.arange(self.rnd_train.shape[0])
        np.random.shuffle(perm)
        self.rnd_train = self.rnd_train[perm]

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        l = self.rnd_train.shape[0]
        if self.batch_offset > l:
            perm = np.arange(l)
            np.random.shuffle(perm)
            self.rnd_train = self.rnd_train[perm]
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        batch = self.rnd_train[start: end]

        # Parse txt file
        batch_x = [self._parse(txt) for txt in batch[:, 0]]
        batch_x = np.array(batch_x)
        batch_y = batch[:, 1].astype(int)
        # gaussin filter
        # y_g = signal.savgol_filter(batch_x, 5, 2)
        # g = Gaussian1DKernel(stddev=5)
        # batch_x = convolve(batch_x, g)

        # low pass filter
        # b, a = signal.butter(2, 0.3, 'low')
        # batch_x = signal.filtfilt(b, a, batch_x)

        batch_x = batch_x/np.mean(batch_x,axis=1).reshape(-1,1)
        return batch_x, batch_y

    def next_batch_val(self, batch_size):
        start = self.batch_offset_val
        self.batch_offset_val += batch_size
        l = self.rnd_valid.shape[0]
        if self.batch_offset_val > l:
            perm = np.arange(l)
            np.random.shuffle(perm)
            self.rnd_valid = self.rnd_valid[perm]
            start = 0
            self.batch_offset_val = batch_size

        end = self.batch_offset_val
        batch = self.rnd_valid[start: end]

        # Parse txt file
        batch_x = [self._parse(txt) for txt in batch[:, 0]]
        batch_x = np.array(batch_x)
        batch_y = batch[:, 1].astype(int)
        #
        # y_g = signal.savgol_filter(batch_x, 5, 2)
        # g = Gaussian1DKernel(stddev=5)
        # batch_x = convolve(batch_x, g)

        # low pass filter
        # b, a = signal.butter(2, 0.3, 'low')
        # batch_x = signal.filtfilt(b, a, batch_x)

        batch_x = batch_x / np.mean(batch_x, axis=1).reshape(-1, 1)
        return batch_x, batch_y


