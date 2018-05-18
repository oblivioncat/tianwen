import pandas as pd
import numpy as np
import os

path = './vote/'

files = os.listdir(path)
print(files)
files.remove('.DS_Store')
new_res = []
for i,f in enumerate(files):
    predict = pd.read_csv(path + f, header=0)
    # print(predict.values[0,:])
    if i == 0:
        for index in range(len(predict)):
            cur_prob = predict.values[index,1:]
            new_res.append(list(cur_prob))
    else:
        new_res = np.array(new_res)
        for index in range(len(predict)):
            old_prob = new_res[index,:]
            cur_prob = predict.values[index,1:]
            new_prob = old_prob+cur_prob
            new_res[index,:]=new_prob

predict = pd.read_csv(path + files[0], header=0)
id_list = predict.values[:,0].astype(int)

max_idx = np.argmax(new_res,axis=1)
map = {0:'star', 1:'unknown', 2:'galaxy', 3:'qso'}
class_name = [map[max_idx[i]] for i in range(len(max_idx))]

class_name= np.expand_dims(class_name, axis=-1)
id_list = np.expand_dims(id_list, axis=-1)
df = np.concatenate((id_list, class_name), axis=-1)
df = pd.DataFrame(df)
df.to_csv('./voted9.csv', index=False)

# df = pd.read_csv('./submit_20180323_103454.csv',header=None)
name = ['star','unknown','galaxy','qso']
distrib = df.loc[:,1]
print(distrib.value_counts())
