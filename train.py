import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import math
from math  import radians, cos, sin, sqrt, asin
import warnings
from utils.w2g import geohash_encode
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
warnings.filterwarnings('ignore')

def get_data(data, mode='train'):

    assert mode=='train' or mode=='test'
    if mode=='train':
        data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True)
    elif mode=='test':
        data['temp_timestamp'] = data['timestamp']
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    data['lon'] = data['longitude'].astype(float)
    data['loadingOrder'] = data['loadingOrder'].astype(str)
    data['lat'] = data['latitude'].astype(float)
    data['speed'] = data['speed'].astype(float)
    data['direction'] = data['direction'].astype(float)
    return data   

class MODEL(object):

    def mse_score_eval(self, preds, valid):
        labels = valid.get_label()
        scores = mean_squared_error(y_true=labels // 3600, y_pred=preds // 3600) 
        return 'mse_score', scores, True

    def build_model(self, train, test, feature, label, seed=1024, is_shuffle=True):
        train_pred = np.zeros((train.shape[0], ))
        test_pred = np.zeros((test.shape[0], ))
        n_splits = 5
        # Kfold
        fold = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
        kf_way = fold.split(train[feature])
        # params
        params = {
            'learning_rate': 1,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'num_leaves': 32,
            # 'feature_fraction': 0.8,
            # 'min_child_samples': 6, 
            # 'bagging_fraction': 0.8,
            # 'min_data_in_leaf': 10,
            'nthread': 8,
            'verbose': 1,
        }
        # train
        for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
            train_x, train_y = train[feature].iloc[train_idx], train[label].iloc[train_idx]
            valid_x, valid_y = train[feature].iloc[valid_idx], train[label].iloc[valid_idx]
            # 数据加载
            n_train = lgb.Dataset(train_x, label=train_y)
            n_valid = lgb.Dataset(valid_x, label=valid_y)

            clf = lgb.train(params=params,
                train_set=n_train,
                num_boost_round=3000,
                valid_sets=[n_valid],
                early_stopping_rounds=100,
                verbose_eval=100,
                feval=self.mse_score_eval
            )
            train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
            test_pred += clf.predict(test[feature], num_iteration=clf.best_iteration)/fold.n_splits
            
        test['label'] = test_pred
        print(test_pred)
        return test[['loadingOrder', 'label']]


if __name__ == "__main__":
    TRAIN_GPS_PATH= '/data4/mjx/GPS/train.csv'
    
    ORDER_DATA_PATH = '/data4/mjx/GPS/loadingOrderEvent.csv'
    PORT_DATA_PATH = '/data4/mjx/GPS/port.csv'
  
    TEST1_DATA_PATH = 'test_1.csv'
    TEST2_DATA_PATH = 'test_2.csv'
    TEST3_DATA_PATH = 'test_3.csv'

    test_data = pd.read_csv(TEST2_DATA_PATH)
    test_data = get_data(test_data, mode='test')


    test = pd.read_csv('test_EDA2.csv')
    train  = pd.read_csv('train_EDA2.csv')
    features = [c for c in train.columns if c not in ['loadingOrder', 'label']]
    model = MODEL()
    result = model.build_model(train, test, features, 'label', is_shuffle=True)
    test_data = test_data.merge(result, on='loadingOrder', how='left')
    res_csv = pd.read_csv('trainEDAn2.csv')
    yy = {}
    for col in res_csv.columns[4:]:
        count = 0
        summ = 0
        for num in res_csv[col].iloc[1:]:
            if num != 0:
                count += 1
                summ += num
        if count == 0:
            pass
        else:
            yy[col] = summ /count
    # print(yy)
    for i in range(len(test_data)):
        try:
            if test_data['TRANSPORT_TRACE'].iloc[i] in yy.keys(): 
                test_data.replace(test_data['label'].iloc[i], yy[test_data['TRANSPORT_TRACE'].iloc[i]],inplace=True)
        except:
            print(test_data['TRANSPORT_TRACE'].iloc[i])
            
    test_data.to_csv('11.csv')
    test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x:pd.Timedelta(seconds=x))).apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data.drop(['direction','TRANSPORT_TRACE'],axis=1,inplace=True)
    test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
    test_data['timestamp'] = test_data['temp_timestamp']
    # 整理columns顺序
    result = test_data[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]
    result.to_csv('result1.csv', index=False)