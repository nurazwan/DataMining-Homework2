from data_prep import get_data_reduced
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

X_train, X_test, y_train, y_test = get_data_reduced()

F_params={'n_estimators':[100,200,400,800,1600],'criterion':['gini', 'entropy'], 
'max_depth':[2,5,10,None], 'min_samples_split':[2,4,8], 'min_samples_leaf':[2,4,8],
'bootstrap':[True,False]}

Model=GridSearchCV(RandomForestClassifier(),param_grid=F_params,cv=10,verbose=5, n_jobs=-1,scoring='recall')
RF_model=Model.fit(X_train,y_train)

dump(RF_model,'RandomForest_pca.joblib')