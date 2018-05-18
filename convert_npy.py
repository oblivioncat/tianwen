from batch_data_reader import BatchDataset
import numpy as np
import os
from sklearn.preprocessing import scale
import pandas as pd
from utils import preprocessing

test_txt_dir = '../second_a_test_data_20180313'
test_idx_file = './second_a_test_index_20180313.csv'

def _parse_test(file):
    with open(test_txt_dir+os.sep+str(file)+'.txt','r') as f:
        content = f.readline().rstrip('')
    content = map(np.float32, content)
    content = scale(content,axis=-1)

    return content

reader = BatchDataset(valid_size=0.0)
reader.refresh_rnd_train()

train_X, train_y = reader.next_batch(batch_size=433851)

np.save('x.npy', train_X)
np.save('y.npy', train_y)

df = pd.read_csv(test_idx_file).as_matrix()
test_id = np.squeeze(df)

test_X = [_parse_test(id) for id in test_id]
np.save('test_x.npy', np.array(test_X))

# with open('./data/1614542.txt','r') as f:
#     content = f.readline().rstrip('\n').split(',')
# print(content)
# content = map(np.float32,content)
# print(np.array(content))
# content = scale(content,axis=-1)
# print(content)