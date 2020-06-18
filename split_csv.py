import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from shutil import copyfile

TRAIN_DATA_PATH = '/data4/mjx/GPS/train'
TRAIN1_DATA_PATH = '/data4/mjx/GPS/train1'
TRAIN2_DATA_PATH = '/data4/mjx/GPS/train2'
TRAIN3_DATA_PATH = '/data4/mjx/GPS/train3'

TEST_DATA_PATH = '/data4/mjx/GPS/test.csv'
TEST1_DATA_PATH = '/data4/mjx/GPS/test1'
TEST2_DATA_PATH = '/data4/mjx/GPS/test2'
TEST3_DATA_PATH = '/data4/mjx/GPS/test3'

test_5 = ['CNSHK-SGSIN', 'CNSHK-MYTPP', 'CNSHA-SGSIN']
test_30 = ['HKHKG-FRFOS','CNYTN-MTMLA', 'CNSHK-CLVAP', 'COBUN-HKHKG', 'CNYTN-ARENA', 'CNSHK-SIKOP', 'CNSHK-ESALG', 'CNYTN-PAONX', 'CNSHK-GRPIR', 'CNYTN-RTM']


# 切分test.csv
def split_test():
    
    test_data = pd.read_csv('/data4/mjx/GPS/test.csv')
    grouped = test_data.groupby('loadingOrder')
    for i,g in tqdm(grouped):
        g.to_csv('/data4/mjx/GPS/test/{}.csv'.format(i), index=False)
    test_data = pd.read_csv('/data4/mjx/GPS/test.csv')
    test_1 = pd.DataFrame(columns=test_data.columns.values)
    test_2 = pd.DataFrame(columns=test_data.columns.values)
    test_3 = pd.DataFrame(columns=test_data.columns.values)
    for i in tqdm(range(len(test_data))):
        if test_data.iloc[i, -1] in test_5:
            test_1 = test_1.append(test_data.iloc[i, :])
            test_1.to_csv('test_1.csv',index=False)
        elif test_data.iloc[i, -1] in test_30:
            test_3 = test_3.append(test_data.iloc[i, :])
            test_3.to_csv('test_3.csv',index=False)
        else:
            test_2 = test_2.append(test_data.iloc[i, :])
            test_2.to_csv('test_2.csv',index=False)
            

    for i in os.listdir('/data4/mjx/GPS/test'):   
        test_data = pd.read_csv(os.path.join('/data4/mjx/GPS/test', i))
        if  test_data.iloc[1, -1] in test_5:
            copyfile(os.path.join('/data4/mjx/GPS/test', i), os.path.join(TEST1_DATA_PATH, i))
        elif test_data.iloc[1, -1] in test_30:
            copyfile(os.path.join('/data4/mjx/GPS/test', i), os.path.join(TEST3_DATA_PATH, i))
        else:
            copyfile(os.path.join('/data4/mjx/GPS/test', i), os.path.join(TEST2_DATA_PATH, i))

# 组合结果
def contact_result():
    for i in ['result1.csv','result2.csv','result3.csv']:
        x = pd.read_csv(i)
        x.to_csv('result_44.csv', mode='a',index=False, header=False)

# 切分训练集
def split_train():
    train_files = os.listdir(TRAIN_DATA_PATH)
    for i in tqdm(train_files):
        csv_data = pd.read_csv(os.path.join(TRAIN_DATA_PATH, i), low_memory=False)
        csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'], infer_datetime_format=True)
        group_df = csv_data.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
        total_time = float((group_df['mmax'] - group_df['mmin']).dt.total_seconds())
        if total_time > 2100000:
            copyfile(os.path.join(TRAIN_DATA_PATH, i), os.path.join(TRAIN3_DATA_PATH, i))
        if total_time< 2100000 and total_time > 1555200:
            copyfile(os.path.join(TRAIN_DATA_PATH, i), os.path.join(TRAIN2_DATA_PATH, i))
        if total_time < 1555200 and total_time > 1000000:
            copyfile(os.path.join(TRAIN_DATA_PATH, i), os.path.join(TRAIN4_DATA_PATH, i))
        if total_time < 900000 and total_time > 250000:
            copyfile(os.path.join(TRAIN_DATA_PATH, i), os.path.join(TRAIN1_DATA_PATH, i))

      
if __name__=="__main__":
    # split_test()
    # split_train()




