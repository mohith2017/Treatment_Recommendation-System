from sklearn.externals import joblib
import pandas as pd

data_test=pd.read_csv('testms.csv')

data_test['mental_health_interview']=data_test['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})

data_test=data_test[['mental_health_interview']]
X_test=data_test.loc[:, data_test.columns != 'treatment']

# load the model from disk
filename = 'finalized_model.pkl'
loaded_model = joblib.load("C:/Users/Mohith/Desktop/Datasets/models/"+filename)
result = loaded_model.predict(X_test)
print(result)
