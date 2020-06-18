import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from shutil import copyfile
from utils.w2g import geohash_encode
from gensim.models import Word2Vec

TRAIN_GPS_PATH= '/data4/mjx/GPS/train.csv'
TRAIN_DATA_PATH = '/data4/mjx/GPS/train44'
NEW_TRAIN_DATA_PATH = '/data4/mjx/GPS/new_train'
TEST_DATA_PATH = '/data4/mjx/GPS/test'
ORDER_DATA_PATH = '/data4/mjx/GPS/loadingOrderEvent.csv'
PORT_DATA_PATH = '/data4/mjx/GPS/port.csv'

def processing_csv(i):
   
    port_data = pd.read_csv(PORT_DATA_PATH)
    csv_data = pd.read_csv(os.path.join(TRAIN_DATA_PATH, i), low_memory=False)
    
    # 筛掉距离过短的数据， 追踪trace 进行补全, 筛掉没有trace的文件
    csv_data['TRANSPORT_TRACE'] = csv_data['TRANSPORT_TRACE'].fillna(method = 'backfill')
    csv_data['TRANSPORT_TRACE'] = csv_data['TRANSPORT_TRACE'].fillna(method = 'pad')
    if csv_data['TRANSPORT_TRACE'].isnull().all():
        os.remove(os.path.join(TRAIN_DATA_PATH, i))
    # 删除不是直达的
    # if len(str(csv_data['TRANSPORT_TRACE'][0]).split('-')) != 2:
    #     os.remove(os.path.join(TRAIN_DATA_PATH, i))
    # 删除没有从出发港口出发
    leave = str(csv_data['TRANSPORT_TRACE'][0]).split('-')[0]
    leave_lon = float(port_data[port_data.TRANS_NODE_NAME == leave]['LONGITUDE'].iloc[0])
    leave_lat = float(port_data[port_data.TRANS_NODE_NAME == leave]['LATITUDE'].iloc[0])
    if np.abs(csv_data['latitude'].iloc[0] - leave_lat) > 0.1 or np.abs(csv_data['longitude'].iloc[0] - leave_lon) > 0.1:
        os.remove(os.path.join(TRAIN_DATA_PATH, i))
    # 删除没有到达目的地港口
    destination = str(csv_data['TRANSPORT_TRACE'][0]).split('-')[-1]
    des_lon = float(port_data[port_data.TRANS_NODE_NAME == destination]['LONGITUDE'].iloc[0])
    des_lat = float(port_data[port_data.TRANS_NODE_NAME == destination]['LATITUDE'].iloc[0])
    if np.abs(csv_data['latitude'].iloc[-1] - des_lat) > 1 or np.abs(csv_data['longitude'].iloc[-1] - des_lon) > 1:
        os.remove(os.path.join(TRAIN_DATA_PATH, i))
   
    csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'], infer_datetime_format=True)
    # 删除前n行和后n行 速度为0
    max_0 = max(csv_data[csv_data['speed']!=0].index)
    min_0 = min(csv_data[csv_data['speed']!=0].index)
    if (len(csv_data) - max_0) > 10 and min_0 > 10: 
        csv_data = csv_data.iloc[min_0-5:max_0+5, :]
        csv_data.to_csv((os.path.join(TRAIN_DATA_PATH, i)), index=False)
    else:
        csv_data = csv_data.iloc[min_0:max_0, :]
        csv_data.to_csv((os.path.join(TRAIN_DATA_PATH, i)), index=False)

    
    
if __name__ == "__main__":
    p = Pool(48)
    train_files = os.listdir(TRAIN_DATA_PATH)
    for i in tqdm(train_files):
        p.apply_async(processing_csv, args=(i, ))
    p.close()
    p.join()
    