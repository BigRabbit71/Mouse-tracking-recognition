# coding:utf-8
import pandas as pd
import scipy as sp
import numpy as np
import sklearn
import gc
import warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn import metrics, cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer
import lightgbm as lgb
import xgboost as xgb
import matplotlib
import os

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

warnings.filterwarnings("ignore")

cache = '.../Cache'
result = '.../Result'
featureImportanceDir = '.../Feature Importance'
datadir = '.../Data'

train_path = os.path.join(datadir, 'dsjtzs_txfz_training.txt')
test_path = os.path.join(datadir, 'dsjtzs_txfz_test1.txt')
testb_path = os.path.join(datadir, 'dsjtzs_txfz_testB.txt')

if not os.path.exists(cache):
    os.mkdir(cache)
if not os.path.exists(result):
    os.mkdir(result)
if not os.path.exists(featureImportanceDir):
    os.mkdir(featureImportanceDir)

def applyParallel(dfGrouped, func):
    with Parallel(n_jobs=64) as parallel:
        retLst = parallel(delayed(func)(pd.Series(value)) for key, value in dfGrouped)
        return pd.concat(retLst, axis=0)


def draw(df):
    import matplotlib.pyplot as plt
    if not os.path.exists('pic'):
        os.mkdir('pic')

    points = []
    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append((float(point[0]) / 7, float(point[1]) / 13))

    x, y = zip(*points)
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    plt.plot(x, y)
    plt.subplot(122)
    plt.plot(x, y)
    aim = df.aim.split(',')
    aim = (float(aim[0]) / 7, float(aim[1]) / 13)
    plt.scatter(aim[0], aim[1])
    plt.title(df.label)
    plt.savefig('pic/%s-label=%s' % (df.idx, df.label))
    plt.clf()
    plt.close()


def get_feature(df):
    points = []

    # points = [((353.0, 2607.0), 349.0),
    #           ((367.0, 2607.0), 376.0),
    #           .............]
    for point in df.trajectory[:-1].split(';'):
        point = point.split(',')
        points.append(((float(point[0]), float(point[1])), float(point[2])))

    xs = pd.Series([point[0][0] for point in points])
    ys = pd.Series([point[0][1] for point in points])

    aim = df.aim.split(',')
    aim = (float(aim[0]), float(aim[1]))

    # 提取特征部分省略。。。

    return df.to_frame().T

def make_train_set():
    dump_path = os.path.join(cache, 'train.hdf')
    if os.path.exists(dump_path):
        train = pd.read_hdf(dump_path, 'all')
    else:
        train = pd.read_csv(train_path, sep=' ', header=None, names=['id', 'trajectory', 'aim', 'label'])
        train['count'] = train.trajectory.map(lambda x: len(x.split(';')))
        train = applyParallel(train.iterrows(), get_feature).sort_values(by='id')
        # 写入hdf5文件
        train.to_hdf(dump_path, 'all')
    return train


def make_test_set():
    dump_path = os.path.join(cache, 'test.hdf')
    if os.path.exists(dump_path):
        test = pd.read_hdf(dump_path, 'all')
    else:
        test = pd.read_csv(test_path, sep=' ', header=None, names=['id', 'trajectory', 'aim'])
        test['count'] = test.trajectory.map(lambda x: len(x.split(';')))
        test = applyParallel(test.iterrows(), get_feature).sort_values(by='id')
        # 写入hdf5文件
        test.to_hdf(dump_path, 'all')
    return test

def make_testb_set():
    dump_path = os.path.join(cache, 'testb.hdf')
    if os.path.exists(dump_path):
        testb = pd.read_hdf(dump_path, 'all')
    else:
        testb = pd.read_csv(testb_path, sep=' ', header=None, names=['id', 'trajectory', 'aim'])
        testb['count'] = testb.trajectory.map(lambda x: len(x.split(';')))
        testb = applyParallel(testb.iterrows(), get_feature).sort_values(by='id')
        # 写入hdf5文件
        testb.to_hdf(dump_path, 'all')
    return testb

if __name__ == '__main__':
    draw_if = False
    train, test = make_train_set(), make_test_set()
    testb = make_testb_set()
    # train = make_train_set()
    if draw_if:
        train.reset_index().rename(columns={'index': 'idx'}).apply(draw, axis=1)

    # 观察一下label分布
    print(train['label'].value_counts())


    # 得到训练集
    training_data, label = train.drop(['id', 'trajectory', 'aim', 'label'], axis=1).astype(float), train['label']
    # 得到测试集
    sub_training_data, instanceIDs = test.drop(['id', 'trajectory', 'aim'], axis=1).astype(float), test['id']

    sub_training_data_b, instanceIDs_b = testb.drop(['id', 'trajectory', 'aim'], axis=1).astype(float), testb['id']

    # print (training_data.shape)
    #
    # train_x, test_x, train_y, test_y = train_test_split(training_data, label, test_size=0.01, random_state=0)

    # 需要做一下GBDT！！！！！！！！！
    # 调参尝试！！！！！！！！

    # gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_depth=7, min_samples_leaf=20,
    #                                  min_samples_split=20, subsample=0.8)
    gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, min_samples_split=50, max_depth=9,
                                     min_samples_leaf=4, max_features=15, subsample=0.7)

    # 缺失值处理
    training_data = training_data.fillna(0)
    sub_training_data = sub_training_data.fillna(0)
    sub_training_data_b = sub_training_data_b.fillna(0)

    label = label.values.astype(int)

    # train
    print('Start training...')
    gbm.fit(training_data, label)
    label_pred = gbm.predict(training_data)
    label_predprob = gbm.predict_proba(training_data)[:, 1]
    print "Accuracy: %.4f" % metrics.accuracy_score(label, label_pred)
    print "AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob)
    # cross-validation 交叉验证
    cv_score = cross_validation.cross_val_score(gbm, training_data, label, cv=3, scoring='roc_auc')
    print "CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
    np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # predict
    print('Start predicting...')
    y_pred = gbm.predict(sub_training_data)
    y_predprob = gbm.predict_proba(sub_training_data)[:, 1]

    y_pred_b = gbm.predict(sub_training_data_b)
    y_predprob_b = gbm.predict_proba(sub_training_data_b)[:, 1]

    res = instanceIDs.to_frame()
    res['label'] = y_pred
    res['prob'] = y_predprob

    res_b = instanceIDs_b.to_frame()
    res_b['label'] = y_pred_b
    res_b['prob'] = y_predprob_b

    # 观察一下y分布
    print 'predict label count for A:'
    print(res['label'].value_counts())
    print 'predict label count for B:'
    print(res_b['label'].value_counts())

    res['id'] = res['id'].astype(int)
    res_b['id'] = res_b['id'].astype(int)

    # # 得到预测结果为0的样本id
    # res[res.label<1].id.to_csv(os.path.join(sub, '20170712.txt'), header=None, index=False)
    # 按照概率升序（默认）排列
    res = res.sort_values(by='prob', axis='index', ascending=True)
    res_b = res_b.sort_values(by='prob', axis='index', ascending=True)

    # 取res中0~19999行的数据
    res.iloc[0:19500].id.to_csv(os.path.join(result, 'dsjtzs_txfzjh_preliminary_0720_a.txt'), header=None, index=False)
    # Plan A: 取前18000个
    res_b.iloc[0:18000].id.to_csv(os.path.join(result, 'dsjtzs_txfzjh_preliminary_0720_b_planA.txt'), header=None, index=False)

    print res['prob'].values[19999]
    # Plan B: 取概率小于res['prob'].values[19999]的

    prob = res['prob'].values[19999]
    for i in range(100000):
        if(res_b['prob'].values[i] < prob):
          print i

    # i = 20456
    res_b.iloc[0:20457].id.to_csv(os.path.join(result, 'dsjtzs_txfzjh_preliminary_0720_b_planB.txt'), header=None,
                                  index=False)

    print("Done!")
