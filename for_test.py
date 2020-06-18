import pandas as pd
from geopy.distance import geodesic
from gensim.models import Word2Vec
import os
from tqdm import tqdm 
NEW_TRAIN_CSV_PATH = '/data4/mjx/GPS/new_train'



# # print(geodesic((22.450483,113.8942), (22.45065,113.894317)).m)
# # def hashfxn(astring):
# #     return ord(astring[0])
# df = pd.read_csv('res.csv')
# # data_frame = df.groupby('loadingOrder')['trace'].agg(list).reset_index()
# # word_list = [str(i).split('-') for i in data_frame['trace']]
# # print(word_list)
# res = []
# for i in df['trace']:
#     res.append(i.split('-'))
# # print(res)
# # print(data_frame['trace'].values)
# model_wv = Word2Vec(res, size=3, window=3, min_count=1, sg=1, hs=1, workers=1, iter=10, seed=1)


# data_frame['trace'] = data_frame['trace'].apply(lambda x: pd.DataFrame([model_wv[c] for c in x]))
# for m in range(10):
#     data_frame['w2v_{}_mean'.format(m)] = data_frame['trace'].apply(lambda x: x[m].mean())
# del data_frame['trace']
# data_frame.to_csv('res22.csv')
# csv_data = pd.read_csv('222.csv')
# print(len(csv_data.groupby('vesselMMSI')))
# csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'], infer_datetime_format=True)
# group_df = csv_data.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
# print((group_df['mmax'] - group_df['mmin']).dt.total_seconds())
# print(csv_data['timestamp'].first(0).dt.total_seconds())
# print(port_data['TRANS_NODE_NAME'].iloc[-1])
# print(port_data.iloc[-1,0])


# 22种目的地， 6个出发点，
# arr_des = []
# carr_name = []
# import numpy as np
# PORT_DATA_PATH = '/data4/mjx/GPS/port.csv'
# for i in os.listdir('/data4/mjx/GPS/test'):
#     csv_data = pd.read_csv(os.path.join('/data4/mjx/GPS/test', i))
#     port_data = pd.read_csv(PORT_DATA_PATH)
#     leave = str(csv_data['TRANSPORT_TRACE'][0]).split('-')[0]
#     leave_lon = float(port_data[port_data.TRANS_NODE_NAME == leave]['LONGITUDE'].iloc[0])
#     leave_lat = float(port_data[port_data.TRANS_NODE_NAME == leave]['LATITUDE'].iloc[0])
#     if np.abs(csv_data['latitude'].iloc[0] - leave_lat) > 1 or np.abs(csv_data['longitude'].iloc[0] - leave_lon) > 1:
#        print(i)
  
#     arr_des.append(test_data.iloc[1, -1])
#     carr_name.append(test_data.iloc[1, 6])
# arr_des = list(set(arr_des))
# carr_name = list(set(carr_name))
# leave = list(set([ i.split('-')[0] for i in arr_des]))
# destination = list(set([i.split('-')[1] for i in arr_des]))
# print("arr_des:{}, carr_name:{}, leave:{}, dest:{}".format(len(arr_des),len(carr_name), len(leave), len(destination)))
# print(arr_des)
# print(carr_name)

# 切分订单信息
# train_data = pd.read_csv('/data4/mjx/GPS/train.csv', header=None)
# train_data.drop_duplicates(keep='first', inplace=True)
# train_data.columns = ['loadingOrder','carrierName','timestamp','longitude',
#                 'latitude','vesselMMSI','speed','direction','vesselNextport',
#                 'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
# train_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
# grouped = train_data.groupby('loadingOrder')
# for i,g in tqdm(grouped):
#     g.to_csv('/data4/mjx/GPS/train44/{}.csv'.format(i), index=False)

# 将港口编码
# port = {}
# x = []
# port_df = pd.DataFrame(port, columns=["port_name", "port_lat", "port_lon"])
# port_data =  pd.read_csv(PORT_DATA_PATH)
# port_df['port_lat']= port_data['LATITUDE'].astype(float)
# port_df['port_lon'] = port_data['LONGITUDE'].astype(float)
# for i in range(len(port_data['LONGITUDE'])):
#     x.append(geohash_encode(float(port_data['LATITUDE'][i]), float(port_data['LONGITUDE'][i]), 10))
# port_df['port_name'] = x
# port_df.to_csv('port.csv', index=False)
# #提取有效路径
# copyfile(os.path.join(TRAIN_CSV_PATH, i), os.path.join(NEW_TRAIN_CSV_PATH, i)) if len(csv_data)>2000 else  0
# #提取时间差，EDA
# csv_data =  csv_data.drop(csv_data[csv_data["min_diff"]<10].index)
# csv_data =  csv_data.drop(csv_data[csv_data["lon_diff"] <=0.0001 and csv_data["lat_diff"] <= 0.0001].index)
# csv_data.drop_duplicates(subset=['lat','lon'], keep='first', inplace=True)
# csv_data.to_csv(os.path.join(NEW_TRAIN_CSV_PATH, i))
# #对港口数据进行填充
# for name, group in csv_data.groupby('vesselNextport'):
#     if len(group)< 100:
#         csv_data['vesselNextport'].loc[csv_data['vesselNextport']==name] = np.nan
# csv_data['vesselNextport'] = csv_data['vesselNextport'].fillna(method = 'backfill')
# csv_data['vesselNextport'] = csv_data['vesselNextport'].fillna(method = 'pad')
# #寻找中间港口
# port_lattt = []
# port_lonnn = []
# if csv_data['vesselNextport'].isnull().all():
#     port_lattt.append(csv_data['latitude'].iloc[0])
#     port_lonnn.append(csv_data['latitude'].iloc[0])
# else:
#     for _, group in csv_data.groupby('vesselNextport'):
#         port_lattt.append(group['latitude'].iloc[0])
#         port_lonnn.append(group['longitude'].iloc[0])

# group_df = csv_data.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
    # 删除时间小于6000分钟
# if  float((group_df['mmax'] - group_df['mmin']).dt.total_seconds() // 60) < 6000:
#     os.remove(os.path.join('/data4/mjx/GPS/new_train', i))
# def change_csv(path, col):
#     h_data = pd.read_csv(path)
#     NEW_TRAIN_DATA_PATH = '/data4/mjx/GPS_2/train44'
#     h_data = h_data.replace(h_data['TRANSPORT_TRACE'].iloc[0], 'HKHKG-FRFOS')
#     h_data.loc[0:col, :].to_csv(os.path.join(NEW_TRAIN_DATA_PATH, path))
# change_csv('/data4/mjx/GPS_2/train44/6051.csv', 6051)