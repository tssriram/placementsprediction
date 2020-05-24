# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:52:15 2020

@author: Tharun
"""
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
   
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))




import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

train=pd.read_csv("C:/Users/Tarun/Desktop/New folder/totalpred.csv")
#train=train.sample(frac=1)

mapd={'CSE':0.1,'Civil':0.2,'ICE':0.3,'EEE':0.4,'Chem':0.5,'Mech':0.6,'Prod':0.7,'ECE':0.8,'MME':0.9}
train.Department=train.Department.map(mapd)

mapg={'Male':1,'Female':0}
train.Gender=train.Gender.map(mapg)

train.GPA=train.GPA/10

mapf={'Software':0.25,'Management':0.5,'Core':0.75,'Analytics':1}
train.Field=train.Field.map(mapf)

train.Backlogs=train.Backlogs/max(train.Backlogs)

train.mark10=train.mark10/100
train.mark12=train.mark12/100

mapq={'OBC':0.25,'General':0.5,'SC':0.75,'PwD':1}
train.Quota=train.Quota.map(mapq)
train.Package=train.Package/45

train.drop(["Gender", "mark12","Quota","Backlogs"], axis = 1, inplace = True)

xTrain, xTest, yTrain, yTest = train_test_split(train.iloc[:,0:4], train.Package, test_size = 0.25, random_state = 0,shuffle=False)


mlp = MLPRegressor(hidden_layer_sizes=(315,315,315),activation='relu',solver='adam',max_iter=1000,alpha=0.0063,learning_rate='adaptive')
mlp.fit(xTrain,yTrain)
pred=mlp.predict(xTest)

print("Test Error")
print(metrics.r2_score(yTest,pred))



    

