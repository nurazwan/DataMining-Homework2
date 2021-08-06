import os
from data_prep import get_data_reduced
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix

#Create folder and change working directory
os.mkdir('LogisticRegression_reduction')
os.chdir('./LogisticRegression_reduction')
#print(os.getcwd())

X_train, X_test, y_train, y_test = get_data_reduced()

print(X_train.head(2))
print(X_test.head(2))
print(y_train.head(2))
print(y_test.head(2))