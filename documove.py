import pandas as pd
import numpy as np
import os, shutil


train1=pd.read_csv('../first_train_index_20180131.csv')
train2=pd.read_csv('../second_a_train_index_20180313.csv')
train1_dir = '../first_train_data_20180131/'
train2_dir = '../second_a_train_data_20180313/'
new_dir = train2_dir

train_index = train1
train_dir = train2_dir

train_unkn = train_index[train_index['type'].isin(['unknown'])]
train_star = train_index[train_index['type'].isin(['star'])]
train_qso = train_index[train_index['type'].isin(['qso'])]
train_galaxy = train_index[train_index['type'].isin(['galaxy'])]

# print('star')
# print(train_star['id'].values[:10])
# print('unknown')
# print(train_unkn['id'].values[:10])
# print('qso')
# print(train_qso['id'].values[:10])
# print('galaxy')
# print(train_galaxy['id'].values[:10])
#
# def movestar():
#     # print(train_star['id'].values)
#     for i in train_star['id'].values:
#         shutil.copyfile(train_dir+str(i)+'.txt',new_dir + str(i)+'.txt')
#         print "copy %s -> %s"%(train1_dir+str(i)+'.txt',new_dir+'star-' + str(i)+'.txt')
# def moveqso():
#     # print(train_qso['id'].values)
#     for i in train_qso['id'].values:
#         shutil.copyfile(train_dir+str(i)+'.txt',new_dir + str(i)+'.txt')
#         print "copy %s -> %s"%(train1_dir+str(i)+'.txt',new_dir+'qso-' + str(i)+'.txt')
# def movegalaxy():
#     # print(train_galaxy['id'].values)
#     for i in train_galaxy['id'].values:
#         shutil.copyfile(train_dir+str(i)+'.txt',new_dir + str(i)+'.txt')
#         print "copy %s -> %s"%(train1_dir+str(i)+'.txt',new_dir+'galaxy-' +str(i)+'.txt')
# def moveunkn():
#     # print(train_unkn['id'].values)
#     for i in train_unkn['id'].values:
#         shutil.copyfile(train_dir+str(i)+'.txt',new_dir + str(i)+'.txt')
#         # os.remove(new_dir+str(i)+'.txt')
#         print "copy %s -> %s"%(train1_dir+str(i)+'.txt',new_dir+'unkn-' +str(i)+'.txt')
#         # print "remove %s" % (new_dir+str(i)+'.txt')
#
def dele():
    for i in train_unkn['id'].values:
        if os._exists(new_dir + str(i)+'.txt'):
            shutil.rmtree(new_dir + str(i)+'.txt')
            # os.remove(new_dir+str(i)+'.txt')
            print "del %s "%(new_dir+str(i)+'.txt')
            # print "remove %s" % (new_dir+str(i)+'.txt')
dele()