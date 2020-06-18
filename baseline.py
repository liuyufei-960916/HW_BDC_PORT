import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import math
from math  import radians, cos, sin, sqrt, asin
import warnings
warnings.filterwarnings('ignore')
train_gps_path = '/data4/mjx/GPS/train.csv'
test_data_path = '/data4/mjx/GPS/A_testData0531.csv'
order_data_path = '/data4/mjx/GPS/loadingOrderEvent.csv'
port_data_path = '/data4/mjx/GPS/port.csv'
# 取前1000000行
debug = True
NDATA = 1000000

if debug:
    train_data = pd.read_csv(train_gps_path,nrows=NDATA,header=None)
else:
    train_data = pd.read_csv(train_gps_path,header=None)

train_data.columns = ['loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
test_data = pd.read_csv(test_data_path)

def get_data(data, mode='train'):
    assert mode=='train' or mode=='test'
    if mode=='train':
        data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True)
    elif mode=='test':
        data['temp_timestamp'] = data['timestamp']
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    data['longitude'] = data['longitude'].astype(float)
    data['loadingOrder'] = data['loadingOrder'].astype(str)
    data['latitude'] = data['latitude'].astype(float)
    data['speed'] = data['speed'].astype(float)
    data['direction'] = data['direction'].astype(float)
    return data

train_data = get_data(train_data, mode='train')
test_data = get_data(test_data, mode='test')
test_data.to_csv('test_data.csv', index = False)
train_data.to_csv('train_data.csv',index = False)
def lon2x (lon):
    """
    :param lon: 经度
    :return:
    """
    L = 6381372*math.pi*2    #地球周长
    W = L                    #平面展开，将周长视为X轴
    x = lon*math.pi/180      #将经度从度数转换为弧度
    x = (W/2)+(W/(2*math.pi))*x
    return round(x)

def lat2y (lat):
    """
    :param lat: 维度
    :return:
    """
    L = 6381372*math.pi*2                      
    H = L/2                  
    mill = 2.3               
    y = lat*math.pi/180      
    y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y))  #米勒投影的转换 
    y = (H/2)-(H/(2*mill))*y  # 这里将弧度转为实际距离 ，转换结果的单位是公里
    return round(y)

#python计算两点间距离-m

def geodistance(lon1,lat1,lon2,lat2):
    lon1 = list(map(np.radians, lon1))
    lon2 = list(map(np.radians, lon2))
    lat1 = list(map(np.radians, lat1))
    lat2 = list(map(np.radians, lat2))
    # lon1, lat1, log2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon=lon2-lon1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    dis=2*asin(sqrt(a))*6371*1000
    return dis


def get_feature(df, mode='train'):
    assert mode=='train' or mode=='test'
    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    # 特征只选择经纬度、速度\方向
    # df['coordinate_x'] = df[['latitude', 'longitude']].apply(lambda x: ','.join(x), axis=1)
    df['coordinate_y'] = df['latitude'].apply(lat2y)
    df['coordinate_x'] = df['longitude'].apply(lon2x)

    # lat_1 = np.array(df.groupby('loadingOrder', as_index = True)['latitude'])
    # lon_1 = np.array(df.groupby('loadingOrder', as_index = True)['longitude'])
    # lat_2 = np.array(df.groupby('loadingOrder')['latitude'].shift(1))
    # lon_2 = np.array(df.groupby('loadingOrder')['longitude'].shift(1))
    # lon_1_1 = map(radians, lon_1)
    # dis = geodistance(lon_1,lat_1,lon_2,lat_2)
    x_diff = np.array(df['coordinate_x'].diff(1))
    y_diff = np.array(df['coordinate_y'].diff(1))
    assert len(x_diff) == len(y_diff)
    diff = np.sqrt(np.square(x_diff) + np.square(y_diff))
    df['distance'] = diff
    # df['coordinate'] = df.groupby('loadingOrder')['latitude'] + df.groupby('loadingOrder')['longitude']
    # lat = df.groupby('loadingOrder').get_group('loadingOrder')
    # a,b  = (df['coordinate'][1]).split(',')[0], (df['coordinate'][1]).split(',')[1]
    

    df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
    df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
    df['speed_diff'] = df.groupby('loadingOrder')['speed'].diff(1)
    df['diff_minutes'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
    df['anchor'] = df.apply(lambda x: 1 if x['lat_diff'] <= 0.03 and x['lon_diff'] <= 0.03
                            and x['speed_diff'] <= 0.3 and x['diff_minutes'] <= 10 else 0, axis=1)
    
    if mode=='train':
        group_df = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
        # 读取数据的最大值-最小值，即确认时间间隔为label
        group_df['label'] = (group_df['mmax'] - group_df['mmin']).dt.total_seconds()
    elif mode=='test':
         group_df = df.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()
    
    
    anchor_df = df.groupby('loadingOrder')['anchor'].agg('sum').reset_index()
    anchor_df.columns = ['loadingOrder', 'anchor_cnt']
    group_df = group_df.merge(anchor_df, on='loadingOrder', how='left')
    group_df['anchor_ratio'] = group_df['anchor_cnt'] / group_df['count']
  

    distance_df = df.groupby('loadingOrder')['distance'].agg('sum').reset_index()
    distance_df.columns = ['loadingOrder', 'distance'] 
    group_df = group_df.merge(distance_df, on='loadingOrder', how='left')
    group_df['distance'] = group_df['distance'] / 10000

    agg_function = ['min', 'max', 'mean', 'median']
    agg_col = ['latitude', 'longitude', 'speed', 'direction']

    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
    group_df = group_df.merge(group, on='loadingOrder', how='left')

    return group_df
  
    
# train = get_feature(train_data, mode='train')
# train.to_csv('train_after_EDA.csv', index=False)
# test = get_feature(test_data, mode='test')
# test.to_csv('test_after_EDA.csv', index=False)
train  = pd.read_csv('train_after_EDA.csv')
test = pd.read_csv('test_after_EDA.csv')
features = [c for c in train.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'count']]

def mse_score_eval(preds, valid):
    labels = valid.get_label()
    scores = mean_squared_error(y_true=labels, y_pred=preds)
    return 'mse_score', scores, True

def build_model(train, test, pred, label, seed=1080, is_shuffle=True):
    train_pred = np.zeros((train.shape[0], ))
    test_pred = np.zeros((test.shape[0], ))
    n_splits = 10
    # Kfold
    fold = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
    kf_way = fold.split(train[pred])
    # params
    params = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 36,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 6,
        'seed': 8,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': 8,
        'verbose': 1,
    }
    # train
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        train_x, train_y = train[pred].iloc[train_idx], train[label].iloc[train_idx]
        valid_x, valid_y = train[pred].iloc[valid_idx], train[label].iloc[valid_idx]
        # 数据加载
        n_train = lgb.Dataset(train_x, label=train_y)
        n_valid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(  params=params,
            train_set=n_train,
            num_boost_round=3000,
            valid_sets=[n_valid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=mse_score_eval
        )
        train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        test_pred += clf.predict(test[pred], num_iteration=clf.best_iteration)/fold.n_splits
    
    test['label'] = test_pred
    
    return test[['loadingOrder', 'label']]

def train_xgb():
    
result = build_model(train, test, features, 'label', is_shuffle=True)
test_data = test_data.merge(result, on='loadingOrder', how='left')
test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x:pd.Timedelta(seconds=x))).apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data.drop(['direction','TRANSPORT_TRACE'],axis=1,inplace=True)
test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
test_data['timestamp'] = test_data['temp_timestamp']
# 整理columns顺序
result = test_data[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]
result.to_csv('result.csv', index=False)