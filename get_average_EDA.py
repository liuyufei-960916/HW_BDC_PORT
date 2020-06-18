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


NEW_TRAIN_DATA_PATH = '/data4/mjx/GPS_2/train44'

PORT_DATA_PATH = '/data4/mjx/GPS/port.csv'
TRAIN_DATA_PATH = '/data4/mjx/GPS/train'

arr_des = list(pd.read_csv('arr_des.csv')['0'])


def processing_csv(i):
    try:
        port_data = pd.read_csv(PORT_DATA_PATH)
        csv_data = pd.read_csv(os.path.join(NEW_TRAIN_DATA_PATH, i), low_memory=False)

        csv_data['timestamp'] = pd.to_datetime(csv_data['timestamp'], infer_datetime_format=True)
        group_df = csv_data.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
        csv_data['loadingOrder'] = csv_data['loadingOrder'].astype(str)
        return_feature = {}
        return_feature['loadingOrder'] = str(i).split('.')[0]
        return_feature['trace'] = str(csv_data['TRANSPORT_TRACE'][0])
        return_feature['trace1'], return_feature['trace2'] = 0, 0
        for ii in arr_des:  
            return_feature[ii] = 0

        path = str(csv_data['TRANSPORT_TRACE'][0]).split('-')[0] + '-' + str(csv_data['TRANSPORT_TRACE'][0]).split('-')[-1]
        if path in arr_des:
            return_feature[path] = float((group_df['mmax'] - group_df['mmin']).dt.total_seconds())

        # if len(str(csv_data['TRANSPORT_TRACE'][0]).split('-')) > 2:
        #     path_1 = str(csv_data['TRANSPORT_TRACE'][0]).split('-')[0] + '-' + str(csv_data['TRANSPORT_TRACE'][0]).split('-')[1]
        #     path_2 = str(csv_data['TRANSPORT_TRACE'][0]).split('-')[0] + '-' + str(csv_data['TRANSPORT_TRACE'][0]).split('-')[2]
        #     if path_1 in arr_des:
        #         path_index = []
        #         return_feature['trace1']  = path_1
        #         for _, group in csv_data.groupby('vesselNextport'):
        #             path_index.append(group.index[0])
        #         path_index = np.sort(path_index)
        #         csv_data = csv_data.replace(csv_data['TRANSPORT_TRACE'].iloc[0],path_1)
        #         # time_delta_1 = (csv_data['timestamp'].iloc[path_index[1]-1] - csv_data['timestamp'].iloc[path_index[0]])
        #         csv_data.loc[0:path_index[1]-1, :].to_csv(os.path.join(NEW_TRAIN_DATA_PATH, str(i).split('.')[0]+'path_1.csv'))
        #         # # print(i , time_delta_1,  time_delta_1.days * 86400 + time_delta_1.seconds , path_index[1], csv_data['timestamp'].iloc[path_index[1]])
        #         # return_feature[path_1] = time_delta_1.days * 86400 + time_delta_1.seconds       
        #     if path_2 in arr_des:
        #         path_index = []
        #         return_feature['trace2']  = path_2
        #         for _, group in csv_data.groupby('vesselNextport'):
        #             path_index.append(group.index[0])
        #         path_index = np.sort(path_index)
        #         csv_data = csv_data.replace(csv_data['TRANSPORT_TRACE'].iloc[0], path_2)
        #         csv_data.loc[0:path_index[2]-1, :].to_csv(os.path.join(NEW_TRAIN_DATA_PATH, str(i).split('.')[0]+'path_2.csv'))
                # time_delta_2 = (csv_data['timestamp'].iloc[path_index[2]-1] - csv_data['timestamp'].iloc[int(path_index[0])])
                # # print(i, time_delta_2, time_delta_2.days * 86400 + time_delta_2.seconds)
                # return_feature[path_2] = time_delta_2.days * 86400 + time_delta_2.seconds
        return return_feature 
    except:
        # print(i)
        os.remove(os.path.join(NEW_TRAIN_DATA_PATH, i))
     

if __name__ == "__main__":
    p = Pool(48)
    res = []
    train_files = os.listdir(NEW_TRAIN_DATA_PATH)
    for i in tqdm(train_files):
        # processing_csv(i)
        # p.apply_async(processing_csv, args=(i, ))
        re = p.apply_async(processing_csv, args=(i, ))
        res.append(re.get())
    p.close()
    p.join()
    res_df = pd.DataFrame(res)
    res_df.to_csv('trainEDAn2.csv', index=False)
