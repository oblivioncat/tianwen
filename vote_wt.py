import pandas as pd
import numpy as np
import os

path = ['./result2','./result3','./result5','./result7', './result_incepres6']

# a voting script with adding weights to results

def sum_log(path,epochs,weight=0):
    """

    :param path:
    :param epochs:
    :return: logits array with size of 10,000 * 4
    """
    files = os.listdir(path)
    print(files)
    if '.DS_Store' in files:
        files.remove('.DS_Store')
    new_res = []
    file_list = []

    if weight==0:
        weight = len(epochs)

    for file in files:
        a = file.split('.')[0].split('_')[-1]
        b = file.split('.')[0].split('_')[1]
        if a == 'lgt' and b in epochs:
            print(path+' : '+file.split('_')[1].split('.')[0])
            file_list.append(file)

    for i,f in enumerate(file_list):
        predict = pd.read_csv(path + '/' + f, header=0)
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
    new_res = np.array(new_res)
    new_res = new_res/weight
    return new_res

log = np.zeros([100000,4])

for p in path[:3]:
    epochs = ['ep11','ep12','ep13','ep14','ep15']
    log = log+sum_log(p,epochs,0)

log = log+sum_log(path[3],['ep11','ep12','ep13','ep14','ep15'],weight=0)


log = log+sum_log(path[4],['ep4','ep5'],weight=0)

predict = pd.read_csv('cnn2_ep1_lgt.csv', header=0)
id_list = predict.values[:,0].astype(int)

max_idx = np.argmax(log,axis=1)
map = {0:'star', 1:'unknown', 2:'galaxy', 3:'qso'}
class_name = [map[max_idx[i]] for i in range(len(max_idx))]

class_name= np.expand_dims(class_name, axis=-1)
id_list = np.expand_dims(id_list, axis=-1)
df = np.concatenate((id_list, class_name), axis=-1)
df = pd.DataFrame(df)
df.to_csv('./vote_wt.csv', index=False)

# df = pd.read_csv('./submit_20180323_103454.csv',header=None)
name = ['star','unknown','galaxy','qso']
distrib = df.loc[:,1]
print(distrib.value_counts())
