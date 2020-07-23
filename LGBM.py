#LGBM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
# Importing the dataset
data = pd.read_csv('trainms.csv')
data_test = pd.read_csv('testms.csv')


data=data.dropna(how='any',axis=0)
data['treatment']=data['treatment'].map({'Yes':1,'No':0})
# data['self_employed']=data['self_employed'].map({'Yes':1,'No':0})
data['family_history']=data['family_history'].map({'Yes':1,'No':0})
data['work_interfere']=data['work_interfere'].map({'Often':1,'Sometimes':0.8,'Never':0,'Rarely':0.4})
data['work_interfere'].fillna(0,inplace=True)
data['tech_company']=data['tech_company'].map({'Yes':1,'No':0})
# data['benefits']=data['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})
data['care_options']=data['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})
# data['wellness_program']=data['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})
# data['seek_help']=data['seek_help'].map({'Yes':1,'No':0,"Don't know":0.5})
# data['anonymity']=data['anonymity'].map({'Yes':1,'No':0,"Don't know":0.5})
data['leave']=data['leave'].map({'Somewhat difficult':0.8,'Somewhat easy':0.6,"Don't know":0.4,'Very difficult':1,'Very easy':0})
# data['mental_health_consequence']=data['mental_health_consequence'].map({'Yes':1,'No':0,'Maybe':0.5})
data['phys_health_consequence']=data['phys_health_consequence'].map({'Yes':1,'No':0,'Maybe':0.5})
data['coworkers']=data['coworkers'].map({'Yes':1,'No':0,'Some of them':0.5})
data['supervisor']=data['supervisor'].map({'Yes':1,'No':0,'Some of them':0.5})
# data['mental_health_interview']=data['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
data['phys_health_interview']=data['phys_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
# data['mental_vs_physical']=data['mental_vs_physical'].map({'Yes':1,'No':0,"Don't know":0.5})
# data['obs_consequence']=data['obs_consequence'].map({'Yes':1,'No':0})

X = data.iloc[:, [7,9,12,14,18,20,21,22,24]].values
y = data.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10
clf = lgb.train(params, d_train, 100)


#Prediction
y_pred=clf.predict(x_test)
#convert into binary values
for i in range(y_pred.size):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:
       y_pred[i]=0

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)


data_test['family_history']=data_test['family_history'].map({'Yes':1,'No':0})
data_test['work_interfere']=data_test['work_interfere'].map({'Often':1,'Sometimes':0.8,'Never':0,'Rarely':0.4})
data_test['work_interfere'].fillna(0,inplace=True)
data_test['tech_company']=data_test['tech_company'].map({'Yes':1,'No':0})
# data['benefits']=data['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})
data_test['care_options']=data_test['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})
# data['wellness_program']=data['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})
# data['seek_help']=data['seek_help'].map({'Yes':1,'No':0,"Don't know":0.5})
# data['anonymity']=data['anonymity'].map({'Yes':1,'No':0,"Don't know":0.5})
data_test['leave']=data_test['leave'].map({'Somewhat difficult':0.8,'Somewhat easy':0.6,"Don't know":0.4,'Very difficult':1,'Very easy':0})
# data['mental_health_consequence']=data['mental_health_consequence'].map({'Yes':1,'No':0,'Maybe':0.5})
data_test['phys_health_consequence']=data_test['phys_health_consequence'].map({'Yes':1,'No':0,'Maybe':0.5})
data_test['coworkers']=data_test['coworkers'].map({'Yes':1,'No':0,'Some of them':0.5})
data_test['supervisor']=data_test['supervisor'].map({'Yes':1,'No':0,'Some of them':0.5})
# data['mental_health_interview']=data['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
data_test['phys_health_interview']=data_test['phys_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})

x_test=data_test.iloc[:, [7,8,11,13,17,19,20,21,23]].values

print(x_test)
#Prediction
y_pred=clf.predict(x_test)
print(y_pred)
#convert into binary values
for i in range(0,259):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:
       y_pred[i]=0

y_pred=np.reshape(y_pred,(259,1),order='C')
print(y_pred.shape)

# #Predicts only for one value
# one_array=np.array([0,0])
# one_array=np.reshape(one_array,(-1,1),order='C').T
# one_value=logreg.predict(np.array([one_array[0][0]]).reshape((-1,1),order='C'))
# print("One value prediction is:",one_value)


my_df = pd.DataFrame(y_pred)
my_df.to_csv('output_test113.csv', index=False)


# save the model to disk
# filename = 'LGBM.pkl'
# joblib.dump(lgb, filename)
# # my_df['treatment']=my_df['treatment'].map({1:'Yes',0:'No'})
# # data_test['treatment']=my_df['treatment']
# #
# # data
#
# loaded_model = joblib.load(filename)
# result = loaded_model.predict(x_test)
# print(result)
