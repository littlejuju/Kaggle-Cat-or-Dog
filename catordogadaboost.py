# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 18:34:09 2019

@author: littlejuju
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import cross_val_score
import scipy.io as sio
import pandas as pd
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
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split=100, min_samples_leaf=5),
                             algorithm="SAMME",
                             n_estimators=10, learning_rate=0.1)
print("check2 pass")
error = [0.]*10 
totaltime = [0.]*10
for t in range(10):
    time_start=time.time()
    bdt.fit(dtrain, labels)
    print("check3 pass")
    datareal = sio.loadmat("test25000.mat")
    dreal = pd.DataFrame(datareal["dogvscat"])
    dreal=dreal.as_matrix()
    dreal=dreal.T
    realcatlabel=np.zeros(12500)
    realdoglabel=np.ones(12500)
    reallabels=np.append(realcatlabel,realdoglabel)
    print("check4 pass")
    predictlabel = bdt.predict(dreal)
    time_end=time.time()
    totaltime[t] = time_end-time_start
    error[t] = len(np.argwhere(reallabels!=predictlabel))/25000
    print('error',error[t])
    print('totally cost',totaltime[t])
    
    
print("error mean:",np.mean(error))
print("error std:",np.std(error))
print("time mean:",np.mean(totaltime))
print("time std:",np.std(totaltime))

#cv part
#bdt.fit(dtrain, labels)
#print("check3 pass")
#datareal = sio.loadmat("dogvscat.mat")
#dreal = pd.DataFrame(datareal["dogvscat"])
#dreal=dreal.as_matrix()
#dreal=dreal.T
#realcatlabel=np.zeros(12500)
#realdoglabel=np.ones(12500)
#reallabels=np.append(realcatlabel,realdoglabel)
#print("check4 pass")
#predictlabel = bdt.predict(dreal)
#    #time_end=time.time()
#scores = cross_val_score(bdt, dtrain, labels) 
#print((1-scores).mean())
#print((1-scores.std())
import winsound
winsound.Beep(600,1000)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    