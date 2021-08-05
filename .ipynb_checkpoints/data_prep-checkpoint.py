def get_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df=pd.read_excel('homework_2_data.xlsx',header=1,index_col='ID')
    df=df.reset_index().drop(columns=['ID'])
    y=df.iloc[:,-1]
    X=df.iloc[:,0:(len(df.columns)-1)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
    return X_train, X_test, y_train, y_test

def get_data_reduced():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA

    df=pd.read_excel('homework_2_data.xlsx',header=1,index_col='ID')
    df=df.reset_index().drop(columns=['ID'])
    
    cat_var=df.iloc[:,1:4]
    num_var=pd.concat([df.LIMIT_BAL,df.iloc[:,5:23]],axis=1)
    pca=PCA(0.95)
    pca_var=pca.fit(num_var)
    pca_var_mod=pca.transform(num_var)
    pca_model=pd.DataFrame(pca_var_mod,columns=['pc1','pc2','pc3','pc4'])

    X=pd.concat([cat_var,pca_model],axis=1)
    y=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
    return X_train, X_test, y_train, y_test