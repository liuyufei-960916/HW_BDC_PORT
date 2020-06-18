import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from shutil import copyfile
from utils.w2g import geohash_encode
from gensim.models import Word2Vec
import math
from math  import radians, cos, sin, sqrt, asin

TRAIN_GPS_PATH= '/data4/mjx/GPS/train.csv'
NEW_TRAIN_DATA_PATH = '/data4/mjx/GPS/new_train'
ORDER_DATA_PATH = '/data4/mjx/GPS/loadingOrderEvent.csv'
PORT_DATA_PATH = '/data4/mjx/GPS/port.csv'

# TRAIN_DATA_PATH = '/data4/mjx/GPS/train'
TRAIN1_DATA_PATH = '/data4/mjx/GPS/train1'
TRAIN2_DATA_PATH = '/data4/mjx/GPS/train2'
TRAIN3_DATA_PATH = '/data4/mjx/GPS/train3'

# TEST_DATA_PATH = '/data4/mjx/GPS/test'

TEST1_DATA_PATH = '/data4/mjx/GPS/test1'
TEST2_DATA_PATH = '/data4/mjx/GPS/test2'
TEST3_DATA_PATH = '/data4/mjx/GPS/test3'
def hashfxn(astring):
    return ord(astring[0])

def w2v_feat(df, group_id, feat, length):

    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    model_wv = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1, hs=1, workers=1, iter=10, seed=1, hashfxn=hashfxn)
    data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([model_wv[c] for c in x]))
    for m in range(length):
        data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(lambda x: x[m].mean())
    del data_frame[feat]
    return data_frame

# # 经度转化为x
def lon2x(lon):
    """
    :param lon: 经度
    :return:
    """
    L = 6356.755*math.pi*2    #地球周长
    W = L                    #平面展开，将周长视为X轴
    x = lon*math.pi/180      #将经度从度数转换为弧度
    x = (W/2)+(W/(2*math.pi))*x
    return round(x)

# # 纬度转化为y
def lat2y (lat):
    """
    :param lat: 维度
    :return:
    """
    L = 6356.755*math.pi*2                      
    H = L/2                  
    mill = 2.3               
    y = lat*math.pi/180      
    y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y))  #米勒投影的转换 
    y = (H/2)-(H/(2*mill))*y  # 这里将弧度转为实际距离 ，转换结果的单位是公里
    return round(y)
# # 计算距离
def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, atan2, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat1-lat2)/2)**2 + cos(lat1)*cos(lat2)*(sin((lon1-lon2)/2)**2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return 6371 * c

def processing_csv(i, is_train):
    try:
        port_data = pd.read_csv(PORT_DATA_PATH)
        if is_train:
            csv_data = pd.read_csv(os.path.join(TRAIN1_DATA_PATH, i), low_memory=False)
        else:
            csv_data = pd.read_csv(os.path.join(TEST1_DATA_PATH, i), low_memory=False)
        csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'], infer_datetime_format=True)
        csv_data['loadingOrder'] = csv_data['loadingOrder'].astype(str)
        # # 填充数据
        for name, group in csv_data.groupby('vesselNextport'):
            if len(group)< 100:
                csv_data['vesselNextport'].loc[csv_data['vesselNextport']==name] = np.nan
            csv_data['vesselNextport'] = csv_data['vesselNextport'].fillna(method = 'backfill')
            csv_data['vesselNextport'] = csv_data['vesselNextport'].fillna(method = 'pad')
            csv_data.to_csv(os.path.join(TRAIN1_DATA_PATH, i), index=False)
        csv_data['speed'] = csv_data['speed'].astype(float)
        csv_data['direction'] = csv_data['direction'].astype(float)
        # # 计算经纬度差距/时间差
        csv_data['lat_diff'] = csv_data['latitude'].abs().diff(1).round(4)
        csv_data['lon_diff'] = csv_data['longitude'].abs().diff(1).round(4)
        csv_data['time_diff'] = csv_data['timestamp'].diff(1).dt.total_seconds()
        csv_data['longitude'] = csv_data['longitude'].astype(float)
        csv_data['latitude'] = csv_data['latitude'].astype(float)
        # csv_data['lat_lon'] = csv_data['latitude'].map(str) + ',' + csv_data['longitude'].map(str)
        return_feature = {}
        return_feature['loadingOrder'] = str(i).split('.')[0]
        return_feature['trace'] = str(csv_data['TRANSPORT_TRACE'][0]).split('-')
        
        # # 是否换船
        return_feature['change_ship'] = len(csv_data.groupby('vesselMMSI')) - 1
        group_df = csv_data.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
        # return_feature['total_time'] = float((group_df['mmax'] - group_df['mmin']).dt.total_seconds())
        total_time = float((group_df['mmax'] - group_df['mmin']).dt.total_seconds())
        if is_train:
            return_feature['label'] = float((group_df['mmax'] - group_df['mmin']).dt.total_seconds())
        # # 计算经纬度变化总和/平均变化
        return_feature['lat_sum'] = csv_data['lat_diff'].abs().sum()
        return_feature['lon_sum'] = csv_data['lon_diff'].abs().sum()
        return_feature['lat_ave'] = float(return_feature['lat_sum'] / (total_time / 3600))
        return_feature['lon_ave'] = float(return_feature['lon_sum'] / (total_time / 3600 ))
        # # 距离
        csv_data['coordinate_y'] = csv_data['latitude'].abs().apply(lat2y)
        csv_data['coordinate_x'] = csv_data['longitude'].abs().apply(lon2x)
        x_diff = np.array(csv_data['coordinate_x'].diff(1))
        y_diff = np.array(csv_data['coordinate_y'].diff(1))
        assert len(x_diff) == len(y_diff)
        diff = np.nansum(np.sqrt(np.square(x_diff) + np.square(y_diff)))
        return_feature['distance'] = diff
        # # 常规特征
        group_lon = csv_data.groupby('loadingOrder')['longitude'].agg(mmax='max', mmean='mean', mmin='min').reset_index()
        group_lat = csv_data.groupby('loadingOrder')['latitude'].agg(mmax='max', mmean='mean', mmin='min').reset_index()
        group_dir = csv_data.groupby('loadingOrder')['direction'].agg(mmean='mean').reset_index()
        group_spe = csv_data.groupby('loadingOrder')['speed'].agg(mmean='mean').reset_index()
        
        return_feature['lat_min'] =  float(group_lat['mmin'])
        return_feature['lat_max'] =  float(group_lat['mmax'])
        return_feature['lat_mean'] = float(group_lat['mmean'])
        return_feature['lon_max'] =  float(group_lon['mmax'])
        return_feature['lon_min'] =  float(group_lon['mmin'])
        return_feature['lon_mean'] = float(group_lat['mmean'])
        return_feature['dir_mean'] = float(group_dir['mmean'])
        return_feature['spe_mean'] = float(group_spe['mmean'])

        # #出发/到达港口经纬度
        return_feature['leave_lat_now'] = csv_data['latitude'].iloc[0]
        return_feature['leave_lon_now'] = csv_data['longitude'].iloc[0]

        leave = str(csv_data['TRANSPORT_TRACE'][0]).split('-')[0]
        leave_port_lat = float(port_data[port_data.TRANS_NODE_NAME == leave]['LATITUDE'].iloc[0])
        leave_port_lon = float(port_data[port_data.TRANS_NODE_NAME == leave]['LONGITUDE'].iloc[0])
        return_feature['leave_port_lat'] = leave_port_lat
        return_feature['leave_port_lon'] = leave_port_lon

        destination = str(csv_data['TRANSPORT_TRACE'][0]).split('-')[-1]
        des_port_lat = float(port_data[port_data.TRANS_NODE_NAME == destination]['LATITUDE'].iloc[0])
        des_port_lon = float(port_data[port_data.TRANS_NODE_NAME == destination]['LONGITUDE'].iloc[0])
        return_feature['des_port_lat'] = des_port_lat
        return_feature['des_port_lon'] = des_port_lon
        
        return_feature['leave_des_distace'] = haversine(des_port_lat, des_port_lon, leave_port_lat, leave_port_lon)
        return_feature['latt_difff'] = np.abs(leave_port_lat - des_port_lat)
        return_feature['lonn_difff'] = np.abs(leave_port_lon - des_port_lon)
        return_feature['des_lat_now'] = csv_data['latitude'].iloc[-1]
        return_feature['des_lon_now'] = csv_data['longitude'].iloc[-1]
        return_feature['trace'] = model_trace[csv_data['TRANSPORT_TRACE'].iloc[0].split('-')].mean()
        # 行驶时间/行驶时间比例
        run_time = csv_data['time_diff'].iloc[csv_data[csv_data['speed']>2].index].sum()
        anchor_time = csv_data['time_diff'].iloc[csv_data[csv_data['speed']<=2].index].sum()
        return_feature['run_time'] = run_time
        return_feature['anchor_time'] = anchor_time
        return_feature['run_ratio'] = run_time / total_time
        return_feature['anchor_ratio'] = anchor_time / total_time

        # #按比例/分位数 取轨迹
        for i in np.arange(0.1, 1, 0.1):
           return_feature['lat_quantile_{}'.format(int(i*100))] = csv_data['latitude'].quantile(i)
          return_feature['lon_quantile_{}'.format(int(i*100))] = csv_data['longitude'].quantile(i)
        for i in np.arange(0.1, 1, 0.1):
          return_feature['lat_ratio_{}'.format(int(i*100))] = csv_data['latitude'].iloc[int(len(csv_data)*i)]
           return_feature['lon_ratio_{}'.format(int(i*100))] = csv_data['longitude'].iloc[int(len(csv_data)*i)]
        return return_feature 
    except:
        print(i)
        # return i
        # TRAIN_DATA_PATH = '/data4/mjx/GPS/train'
        # copyfile(os.path.join(TRAIN_DATA_PATH, i), os.path.join('data4/mjx/GPS/fuck', i))
        # os.remove(os.path.join(TRAIN_DATA_PATH, i))

if __name__ == "__main__":
    p = Pool(48)
    res = []
    model_trace = Word2Vec.load('trace.model')
    test_files = os.listdir(TEST1_DATA_PATH)
    train_files = os.listdir('/data4/mjx/GPS_2/train')
    
    is_train = True
    for i in tqdm(train_files):
        # p.apply_async(processing_csv, args=(i, is_train, ))
        re = p.apply_async(processing_csv, args=(i, is_train, ))
        res.append(re.get())
    p.close()
    p.join()
    res_df = pd.DataFrame(res)
    res_df.to_csv('trainEDA.csv', index=False)
