import os
from data_prep import get_data_reduced
from output_generator import model_metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#Create folder and change working directory
os.mkdir('RandomForest_reduction')
os.chdir('./RandomForest_reduction')
#print(os.getcwd())

#get split data
X_train, X_test, y_train, y_test = get_data_reduced()

#Hyper Parameter
F_params={'n_estimators':[100,200,400,800,1600],'criterion':['gini', 'entropy'], 
'max_depth':[2,5,10,None], 'min_samples_split':[2,4,8], 'min_samples_leaf':[2,4,8],
'bootstrap':[True,False]}

#Train model with cross validation
gs=GridSearchCV(RandomForestClassifier(),param_grid=F_params,cv=10,verbose=5, n_jobs=-1,scoring='recall')
model=gs.fit(X_train,y_train)

model_metrics(model,X_test,y_test)
