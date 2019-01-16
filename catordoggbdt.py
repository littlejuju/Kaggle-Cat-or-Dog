# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:33:44 2019

@author: littlejuju
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import scipy.io as sio
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

datacat = sio.loadmat("traincat.mat")
datadog = sio.loadmat("traindog.mat")
dcat = pd.DataFrame(datacat["TrainData_cat"])
ddog = pd.DataFrame(datadog["TrainData_dog"])
catlabel=np.zeros(1981)
doglabel=np.ones(1930)
dtrain = pd.concat([dcat, ddog], axis=1)
dtrain=dtrain.as_matrix()
dtrain=dtrain.T
labels=np.append(catlabel,doglabel)
print("check1 pass")
x_train, x_test, y_train, y_test = train_test_split(dtrain, labels)

print("check2 pass")
gbr = GradientBoostingClassifier(n_estimators=10, max_depth=5, min_samples_split=2, learning_rate=0.1)
#cv part
#cv_score = cross_validation.cross_val_score(gbr, dtrain, labels, cv=10, scoring='roc_auc')
#print((1-cv_score).mean())
#print((1-cv_score).std())
print("check3 pass")
error = [0.]*10 
totaltime = [0.]*10
for t in range(10):
    time_start=time.time()
    gbr.fit(x_train, y_train.flatten())
    print("check4 pass")
    datareal = sio.loadmat("test25000.mat")
    dreal = pd.DataFrame(datareal["dogvscat"])
    dreal=dreal.as_matrix()
    dreal=dreal.T
    realcatlabel=np.zeros(12500)
    realdoglabel=np.ones(12500)
    reallabels=np.append(realcatlabel,realdoglabel)
    y_gbr = gbr.predict(dreal)
    time_end=time.time()
    totaltime[t] = time_end-time_start
    error[t] = len(np.argwhere(reallabels!=y_gbr))/25000
    print('error',error[t])
    print('totally cost',totaltime[t])
print("error mean:",np.mean(error))
print("error std:",np.std(error))
print("time mean:",np.mean(totaltime))
print("time std:",np.std(totaltime))
import winsound
winsound.Beep(600,1000)





















