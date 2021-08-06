import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report,plot_roc_curve

def model_metrics(model,X_test,y_test):
    pd.DataFrame(classification_report(y_test,model.predict(X_test),output_dict=True)).to_csv('classification_report.csv')
    pd.DataFrame(confusion_matrix(y_test,model.predict(X_test))).to_csv('confusion_matrix.csv')
    plot_roc_curve(model,X_test,y_test)
    plt.savefig('ROC_curve.png')
    plt.close()