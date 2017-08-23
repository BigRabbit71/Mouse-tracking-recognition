# coding:utf-8
# /Users/xyq/anaconda2/bin/python
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
from collections import Counter

rcParams['figure.figsize'] = 12, 4

warnings.filterwarnings("ignore")

cache = '.../Cache'
result = '.../Result'
featureImportanceDir = '.../Feature Importance'
datadir = '.../Data'

train_path = os.path.join(datadir, 'dsjtzs_txfz_training.txt')
test_path = os.path.join(datadir, 'dsjtzs_txfz_test1.txt')

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

    # 提取特征部分省略
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


if __name__ == '__main__':
    train = make_train_set()


    # 观察一下label分布
    print(train['label'].value_counts())

    # 得到训练集
    training_data, label = train.drop(['id', 'trajectory', 'aim', 'label'], axis=1).astype(float), train['label']

    # 需要做一下GBDT！！！！！！！！！
    # # 默认参数的GBM
    # gbm = GradientBoostingClassifier()
    # 调参得到参数后的GBM
    gbm = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, min_samples_split=50, max_depth=9,
                                     min_samples_leaf=4, max_features=15, subsample=0.7)

    # 缺失值处理
    training_data = training_data.fillna(0)
    # sub_training_data = sub_training_data.fillna(0)
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
    print "CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # print feature importance
    # 哇塞超有用 打开了新世界的大门！
    features = [x for x in train.columns if x not in ['id', 'trajectory', 'aim', 'label']]
    featureImportance = pd.Series(gbm.feature_importances_, features).sort_values(ascending=False)
    plt.figure(figsize=(16, 16))
    featureImportance.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature importance score')
    figSavename = os.path.join(featureImportanceDir, 'Feature Importances.png')
    plt.savefig(figSavename)
    plt.clf()
    plt.close()


    # 调参尝试！！！！！！！！

    # # 弱学习器的最大个数，从20开始，每次加10，测到81
    # # 得到最佳值：50，即50个树是最佳的
    # param_test1 = {'n_estimators':range(20, 101, 10)}
    # gbmsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=7,
    #                                         min_samples_leaf=5, min_samples_split=30, subsample=0.8),
    #                                         param_grid = param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # print('param test 1: n_estimators...')
    # gbmsearch1.fit(training_data, label)
    # label_pred = gbmsearch1.predict(training_data)
    # label_predprob = gbmsearch1.predict_proba(training_data)[:, 1]
    # print "Accuracy: %.4f" % metrics.accuracy_score(label, label_pred)
    # print "AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob)
    #
    # # cross-validation 交叉验证
    # cv_score = cross_validation.cross_val_score(gbmsearch1, training_data, label, cv=3, scoring='roc_auc')
    # print "CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
    # np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    # print gbmsearch1.grid_scores_
    # print gbmsearch1.best_params_
    # print gbmsearch1.best_score_

    # # 树的相关参数调试：
    # # 1.max_depth：从5测到15，每次隔2
    # # 2.min_samples_split：从10测到60，每次隔10
    # # 得到最佳值：'min_samples_split': 50, 'max_depth': 9
    # param_test2 = {'max_depth':range(5, 15, 2), 'min_samples_split':range(10, 61, 10)}
    # gbmsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=50,
    #                                         min_samples_leaf=5, subsample=0.8),
    #                                         param_grid = param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # print('param test 2: max_depth & min_sample_split...')
    # gbmsearch2.fit(training_data, label)
    # label_pred = gbmsearch2.predict(training_data)
    # label_predprob = gbmsearch2.predict_proba(training_data)[:, 1]
    # print "Accuracy: %.4f" % metrics.accuracy_score(label, label_pred)
    # print "AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob)
    #
    # # cross-validation 交叉验证
    # cv_score = cross_validation.cross_val_score(gbmsearch2, training_data, label, cv=3, scoring='roc_auc')
    # print "CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
    # np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    # print gbmsearch2.grid_scores_
    # print gbmsearch2.best_params_
    # print gbmsearch2.best_score_

    # # max_features参数调试：从7测到20，每次隔2
    # # 得到最佳值：15
    # param_test3 = {'max_features':range(7, 20, 2)}
    # gbmsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=50, max_depth=9,
    #                                         min_samples_split=50, min_samples_leaf=5, subsample=0.8),
    #                                         param_grid = param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # print('param test 3: max_features...')
    # gbmsearch3.fit(training_data, label)
    # label_pred = gbmsearch3.predict(training_data)
    # label_predprob = gbmsearch3.predict_proba(training_data)[:, 1]
    # print "Accuracy: %.4f" % metrics.accuracy_score(label, label_pred)
    # print "AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob)
    #
    # # cross-validation 交叉验证
    # cv_score = cross_validation.cross_val_score(gbmsearch3, training_data, label, cv=3, scoring='roc_auc')
    # print "CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
    # np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    # print gbmsearch3.grid_scores_
    # print gbmsearch3.best_params_
    # print gbmsearch3.best_score_

    # # learning_rate参数调试：降到0.05，相应地树增加到100
    # # 结果：有提升
    # gbm4 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, min_samples_split=50, max_depth=9,
    #                                  min_samples_leaf=4, max_features=15, subsample=0.7)
    # print('param test 4: learning_rate...')
    # gbm4.fit(training_data, label)
    # label_pred = gbm4.predict(training_data)
    # label_predprob = gbm4.predict_proba(training_data)[:, 1]
    # print "Accuracy: %.4f" % metrics.accuracy_score(label, label_pred)
    # print "AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob)
    #
    # # cross-validation 交叉验证
    # cv_score = cross_validation.cross_val_score(gbm4, training_data, label, cv=3, scoring='roc_auc')
    # print "CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
    # np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # # subsample参数调试
    # # 得到最佳值：0.7
    # param_test5 = {'subsample':[0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    # gbmsearch5 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.05, n_estimators=100, max_depth=9,
    #                                         min_samples_split=50, min_samples_leaf=5, max_features=15),
    #                                         param_grid = param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # print('param test 5: subsample...')
    # gbmsearch5.fit(training_data, label)
    # label_pred = gbmsearch5.predict(training_data)
    # label_predprob = gbmsearch5.predict_proba(training_data)[:, 1]
    # print "Accuracy: %.4f" % metrics.accuracy_score(label, label_pred)
    # print "AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob)
    #
    # # cross-validation 交叉验证
    # cv_score = cross_validation.cross_val_score(gbmsearch5, training_data, label, cv=3, scoring='roc_auc')
    # print "CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
    # np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    # print gbmsearch5.grid_scores_
    # print gbmsearch5.best_params_
    # print gbmsearch5.best_score_

    # # min_samples_leaf参数调试
    # # 得到最佳值：4
    # param_test6 = {'min_samples_leaf':range(1, 10, 1)}
    # gbmsearch6 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.05, max_depth=9, n_estimators=100, max_features=15,
    #                                         min_samples_split=50, subsample=0.7),
    #                                         param_grid = param_test6, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    # print('param test 6: min_samples_leaf...')
    # gbmsearch6.fit(training_data, label)
    # label_pred = gbmsearch6.predict(training_data)
    # label_predprob = gbmsearch6.predict_proba(training_data)[:, 1]
    # print "Accuracy: %.4f" % metrics.accuracy_score(label, label_pred)
    # print "AUC Score(Train): %f" % metrics.roc_auc_score(label, label_predprob)
    #
    # # cross-validation 交叉验证
    # cv_score = cross_validation.cross_val_score(gbmsearch6, training_data, label, cv=3, scoring='roc_auc')
    # print "CV score: Mean: %.7g | Std: %.7g | Min: %.7g | Max: %.7g" % (
    # np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    # print gbmsearch6.grid_scores_
    # print gbmsearch6.best_params_
    # print gbmsearch6.best_score_





    training_data['predLabel'] = label_pred
    print 'Predict value count: \n '
    print training_data['predLabel'].value_counts()

    print("Done!")
