# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 22:03:13 2019

@author: littlejuju
"""
import time
import numpy as np
import xgboost as xgb
import scipy.io as sio
import pandas as pd
import random

datacat = sio.loadmat("traincat.mat")
datadog = sio.loadmat("traindog.mat")
dcat = pd.DataFrame(datacat["TrainData_cat"])
ddog = pd.DataFrame(datadog["TrainData_dog"])
catlabel=np.zeros(1981)
doglabel=np.ones(1930)
labels=np.append(catlabel,doglabel)
#dTrain = pd.concat([dcat, ddog], axis=1)
#dTrain=dTrain.as_matrix()
#dTrain=pd.DataFrame(dTrain.T)
#dTrain = xgb.DMatrix(dTrain, label=labels)

param={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':5,
    'lambda':1,
    'colsample_bytree':0.5,
    'eta': 0.1,
    'seed':719,
    'nthread':7,
     'silent':1}

#cv part
#print('running cross validation, disable standard deviation display')
## do cross validation, this will print result out as
## [iteration]  metric_name:mean_value
#res = xgb.cv(param, dTrain, num_boost_round=10, nfold=5,
#             metrics={'error'}, seed=0,
#             callbacks=[xgb.callback.print_evaluation(show_stdv=False),
#                        xgb.callback.early_stop(30)])
#print(res)
error = [0.]*10 
totaltime = [0.]*10
for t in range(10):
    temp_cat = dcat
    temp_dog = ddog
    dcat = dcat.sample(frac=1, replace=True, random_state=t, axis=1)
    ddog = ddog.sample(frac=1, replace=True, random_state=t, axis=1)
    labels = np.append(np.zeros(1000),np.ones(1000))
    labelstest=np.append(np.zeros(980),np.ones(929))
    dcat_train = dcat.iloc[:,0:1000]
    ddog_train = ddog.iloc[:,0:1000]
    dcat_test = dcat.iloc[:,1001:1981]
    ddog_test = ddog.iloc[:,1001:1930]
    
    dtrain = pd.concat([dcat_train, ddog_train], axis=1)
    dtrain=dtrain.as_matrix()
    dtrain=pd.DataFrame(dtrain.T)
    dtest= pd.concat([dcat_test, ddog_test], axis=1)
    dtest=dtest.as_matrix()
    dtest=pd.DataFrame(dtest.T)
    dtrain = xgb.DMatrix(dtrain, label=labels)
    dtest = xgb.DMatrix(dtest, label=labelstest)
    time_start=time.time()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    #print("check1 pass")
    
    #print("check2 pass")
    #evallist = [(dtest, 'eval'), (dtrain, 'train')]
    print("check3 pass")
    num_round = 10
    bst = xgb.train(param, dtrain, num_round, evallist)
    print("check4 pass")
    datareal = sio.loadmat("test25000.mat")
    dreal = pd.DataFrame(datareal["dogvscat"])
    dreal=dreal.as_matrix()
    dreal=dreal.T
    dreal=pd.DataFrame(dreal)
    realcatlabel=np.zeros(12500)
    realdoglabel=np.ones(12500)
    reallabels=np.append(realcatlabel,realdoglabel)
    dreal = xgb.DMatrix(dreal)
    ypred = bst.predict(dreal)
    time_end=time.time()
    totaltime[t] = time_end-time_start
    error[t] = len(np.argwhere(abs(reallabels-ypred)>0.5))/25000
    print('error',error[t])
    print('totally cost',totaltime[t])
    dcat = temp_cat
    ddog = temp_dog


#test part :)
#for t in range(10):
#t=1
#dcat.sample(frac=1, replace=True, random_state=t, axis=1)
#ddog.sample(frac=1, replace=True, random_state=t, axis=1)
#labels = np.append(np.zeros(1000),np.ones(1000))
#dcat_train = dcat.iloc[:,0:1000]
#ddog_train = ddog.iloc[:,0:1000]
#dcat_test = dcat.iloc[:,1001:1981]
#ddog_test = ddog.iloc[:,1001:1930]
#dtrain = pd.concat([dcat_train, ddog_train], axis=1)
#dtrain=dtrain.as_matrix()
#dtrain=pd.DataFrame(dtrain.T)
#dtest= pd.concat([dcat_test, ddog_test], axis=1)
#dtest=dtest.as_matrix()
#dtest=pd.DataFrame(dtest.T)
#print("check1 pass")
#dTrain = xgb.DMatrix(dtrain, label=labels)
#print("check2 pass")
#param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
#param['nthread'] = 4
#param['eval_metric'] = 'auc'
#evallist = [(dtest, 'eval'), (dtrain, 'train')]
#print("check3 pass")
#num_round = 10
#bst = xgb.train(param, dTrain, num_round, evallist)
#print("check4 pass")
#bst.save_model('0001.model')
#ypred = bst.predict(dtest)
#xgb.plot_importance(bst)
#xgb.plot_tree(bst, num_trees=2)
#xgb.to_graphviz(bst, num_trees=2)
print("error mean:",np.mean(error))
print("error std:",np.std(error))
print("time mean:",np.mean(totaltime))
print("time std:",np.std(totaltime))
import winsound
winsound.Beep(600,1000)





















#
#
#
#
#
#
#
#
#
#
#
#
