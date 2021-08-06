import os
from data_prep import get_data_reduced
from output_generator import model_metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix

#Create folder and change working directory
os.mkdir('LogisticRegression_reduction')
os.chdir('./LogisticRegression_reduction')
#print(os.getcwd())

X_train, X_test, y_train, y_test = get_data_reduced()

params={'C':[0.001,0.01,0.1,1,2,3],
'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

model=GridSearchCV(LogisticRegression(), param_grid=params, scoring='recall', n_jobs=-1, cv=10, verbose=-1)

model_metrics(model,X_test,y_test)
