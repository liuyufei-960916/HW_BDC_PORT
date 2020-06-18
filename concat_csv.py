# 合并筛选过后的CSV ,从中提取经纬度和trace 
# 对lat, lon， trace进行编码
import pandas as pd
import os
from tqdm import tqdm
from gensim.models import Word2Vec
NEW_TRAIN_CSV_PATH = '/data4/mjx/GPS/new_train'

def hashfxn(astring):
    return ord(astring[0])

def con_csv():
    new_train_list = os.listdir(NEW_TRAIN_CSV_PATH)
    df = pd.read_csv(os.path.join(NEW_TRAIN_CSV_PATH, new_train_list[0]))
    for i in tqdm(range(1, len(new_train_list))):
        df = pd.read_csv(os.path.join(NEW_TRAIN_CSV_PATH, new_train_list[i]), low_memory=False)
        df.to_csv('/data4/mjx/GPS/new_train.csv',index=False, mode='a+')

def w2v_feat(df, group_id, feat, length):
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    model_wv = Word2Vec(data_frame[feat].values, size=length, window=5, min_count=1, sg=1, hs=1, workers=1, iter=10, seed=1, hashfxn=hashfxn)
    model_wv.save('lat_lon.model')
    print('finished')

def get_lat_lon():
    df = pd.read_csv('/data4/mjx/GPS/new_train.csv', low_memory=False)
    df['lat_lon'] = df['lat'].map(str) + ' ' + df['lon'].map(str)
    print('start lat_lon word2vec')
    w2v_feat(df, 'loadingOrder', 'lat_lon', 10)

    
def get_trace():
    df = pd.read_csv('/data4/mjx/GPS/new_train.csv', low_memory=False)
    res = []
    for _, group in df.groupby('loadingOrder'):
        res.append(group['TRANSPORT_TRACE'].iloc[0].split('-'))
    print('start trace word2vec')
    model_wv = Word2Vec(res, size=3, window=3, min_count=1, sg=1, hs=1, workers=1, iter=10, seed=1)
    model_wv.save('trace.model')
    print('finished')


if __name__ == "__main__":
    # con_csv()
    # get_trace()
    # get_lat_lon()
    return_feature = {}
    csv_data = pd.read_csv('YK916918338339.csv')
    return_feature['leave_port_lat'] = csv_data['latitude'].iloc[0]
    return_feature['leave_port_lon'] = csv_data['longitude'].iloc[0]
    return_feature['des_port_lat'] = csv_data['latitude'].iloc[-1]
    return_feature['des_port_lon'] = csv_data['longitude'].iloc[-1]