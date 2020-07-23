# import pandas as pd
# df = pd.read_csv("trainms.csv")
# count=0
#
# df1=df['family_history'].to_frame()
# df2=df['treatment'].to_frame()
#
# print(df1.where(df1.values=='Yes').where(df1.values==df2.values).notna())



# mergedStuff = pd.merge(df1, df2, on=['Requiredfamily_history'], how='inner')
# mergedStuff.head()

# for i,j in ,:
#     if  i==j and i=='Yes':
#         count+=1
#
# print(count)

import pandas as pd
from sklearn.externals import joblib
import numpy as np
from sklearn import preprocessing
# import matplotlib.pyplot as plt
# plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
# import seaborn as sns
# sns.set(style="white")
# sns.set(style="whitegrid", color_codes=True)

data=pd.read_csv('trainms.csv')
data_test=pd.read_csv('testms.csv')
data_output_test=pd.read_csv('samplems.csv')

#############################################3

# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import db
#
# cred = credentials.Certificate("C:/Users/Mohith/Desktop/Datasets/smarttreat-544ae-firebase-adminsdk-z7233-7bc9fa2818.json")
# firebase_admin.initialize_app(cred)
#
#
# ###########################################
#
# doc_ref = db.collection(u'data').document(u'IzsGpdg2otGZv6OsFlvY')
#
# try:
#     doc = doc_ref.get()
#     print(doc)
# except google.cloud.exceptions.NotFound:
#     print(u'No such document!')

# data_output_test['treatment']=data_output_test['treatment'].map({'Yes':1,'No':0})
# data['self_employed']=data['self_employed'].map({'Yes':1,'No':0})
# data['family_history']=data['family_history'].map({'Yes':1,'No':0})
#
# data['work_interfere']=data['work_interfere'].dropna(how='any',axis=0)
# data['remote_work']=data['remote_work'].map({'Yes':1,'No':0})
# data['remote_work']=data['remote_work'].map({'Yes':1,'No':0})
# data['tech_company']=data['tech_company'].map({'Yes':1,'No':0})
# print(data['work_interfere'])
# print(data['treatment'])

# count_no_sub = len(data[data['treatment']==0])
# count_sub = len(data[data['treatment']==1])
# pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
# print("percentage of no treatment is", pct_of_no_sub*100)
# pct_of_sub = count_sub/(count_no_sub+count_sub)
# print("percentage of treatment", pct_of_sub*100)
#
# print(data['treatment'].value_counts())
#
# print(data.groupby('self_employed').mean())

# pd.crosstab(data.self_employed,data.treatment).plot(kind='bar')
# plt.title('Whether treatment is required or not')
# plt.xlabel('self_employed')
# plt.ylabel('Frequency of requirement')
# plt.show()

# plt.bar(data['mental_health_interview'], data['treatment'])
# plt.xlabel('Mental health interview taken?')
# plt.ylabel('Treatment required?')
# # plt.xticks(index, label, fontsize=5, rotation=30)
# plt.title('Requirement of treatment')
# plt.show()

data=data[['mental_health_interview','treatment','self_employed','family_history','work_interfere','remote_work','tech_company','benefits','care_options','wellness_program','seek_help','anonymity','leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','phys_health_interview','mental_vs_physical','obs_consequence']]

data=data.dropna(how='any',axis=0)
data['treatment']=data['treatment'].map({'Yes':1,'No':0})
data['self_employed']=data['self_employed'].map({'Yes':1,'No':0})
data['family_history']=data['family_history'].map({'Yes':1,'No':0})
data['work_interfere']=data['work_interfere'].map({'Often':1,'Sometimes':0.8,'Never':0,'Rarely':0.4})
data['remote_work']=data['remote_work'].map({'Yes':1,'No':0})
data['tech_company']=data['tech_company'].map({'Yes':1,'No':0})
data['benefits']=data['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})
data['care_options']=data['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})
data['wellness_program']=data['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})
data['seek_help']=data['seek_help'].map({'Yes':1,'No':0,"Don't know":0.5})
data['anonymity']=data['anonymity'].map({'Yes':1,'No':0,"Don't know":0.5})
data['leave']=data['leave'].map({'Somewhat difficult':0.8,'Somewhat easy':0.6,"Don't know":0.4,'Very difficult':1,'Very easy':0})
data['mental_health_consequence']=data['mental_health_consequence'].map({'Yes':1,'No':0,'Maybe':0.5})
data['phys_health_consequence']=data['phys_health_consequence'].map({'Yes':1,'No':0,'Maybe':0.5})
data['coworkers']=data['coworkers'].map({'Yes':1,'No':0,'Some of them':0.5})
data['supervisor']=data['supervisor'].map({'Yes':1,'No':0,'Some of them':0.5})
data['mental_health_interview']=data['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
data['phys_health_interview']=data['phys_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
data['mental_vs_physical']=data['mental_vs_physical'].map({'Yes':1,'No':0,"Don't know":0.5})
data['obs_consequence']=data['obs_consequence'].map({'Yes':1,'No':0})
# data_test['mental_health_interview']=data_test['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})

print(data)


X = data.loc[:, data.columns != 'treatment']
y = data.loc[:, data.columns == 'treatment']

print(X)

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
columns = X_train.columns
print(X_train)
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['treatment'])

# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no treatment in oversampled data",len(os_data_y[os_data_y['treatment']==0]))
print("Number of treatment",len(os_data_y[os_data_y['treatment']==1]))
print("Proportion of no treatment data in oversampled data is ",len(os_data_y[os_data_y['treatment']==0])/len(os_data_X))
print("Proportion of treatment data in oversampled data is ",len(os_data_y[os_data_y['treatment']==1])/len(os_data_X))

print
data_final_vars=data.columns.values.tolist()
y=['treatment']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols=['mental_health_interview','self_employed','family_history','work_interfere','remote_work','tech_company','benefits','care_options','wellness_program','seek_help','anonymity','leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','phys_health_interview','mental_vs_physical','obs_consequence']
X=os_data_X[cols]
y=os_data_y['treatment']

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


cols=['family_history','work_interfere','tech_company','care_options','leave',
'phys_health_consequence','coworkers','supervisor','phys_health_interview']
X=os_data_X[cols]
y=os_data_y['treatment']

logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

# print(X,y)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# X_train=data['mental_health_interview']
# y_train=data['treatment']
# print(X_train.shape,"  ",y_train.shape,"  ",X_test.shape,"  ",y_test.shape)
# print(X_train)

#

# y_test=data_output_test.loc[:, data_output_test.columns == 'treatment']

# print(X_test,y_test)
from sklearn.svm import SVC
logreg = LogisticRegression()
logreg.fit(X_train, np.ravel(y_train,order='C'))

# print(X_test.shape,"  ",y_test.shape)
# print(X_test)
y_pred = logreg.predict(X_test)
# y_pred=np.reshape(y_pred,(259,1),order='C')


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

data_test=data_test[['family_history','work_interfere','tech_company','care_options','leave',
'phys_health_consequence','coworkers','supervisor','phys_health_interview']]
# data_test=data_test.dropna(how='any',axis=0)
# data['treatment']=data['treatment'].map({'Yes':1,'No':0})
# data_test['self_employed']=data_test['self_employed'].map({'Yes':1,'No':0})
data_test['family_history']=data_test['family_history'].map({'Yes':1,'No':0})
data_test['work_interfere']=data_test['work_interfere'].map({'Often':1,'Sometimes':0.8,'Never':0,'Rarely':0.4})
data_test['work_interfere'].fillna(0, inplace=True)
# data_test['remote_work']=data_test['remote_work'].map({'Yes':1,'No':0})
data_test['tech_company']=data_test['tech_company'].map({'Yes':1,'No':0})
# data['benefits']=data['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})
data_test['care_options']=data_test['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})
# data['wellness_program']=data['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})
# data_test['seek_help']=data_test['seek_help'].map({'Yes':1,'No':0,"Don't know":0.5})
# data['anonymity']=data['anonymity'].map({'Yes':1,'No':0,"Don't know":0.5})
data_test['leave']=data_test['leave'].map({'Somewhat difficult':0.8,'Somewhat easy':0.6,"Don't know":0.4,'Very difficult':1,'Very easy':0})
# data['mental_health_consequence']=data['mental_health_consequence'].map({'Yes':1,'No':0,'Maybe':0.5})
data_test['phys_health_consequence']=data_test['phys_health_consequence'].map({'Yes':1,'No':0,'Maybe':0.5})
data_test['coworkers']=data_test['coworkers'].map({'Yes':1,'No':0,'Some of them':0.5})
data_test['supervisor']=data_test['supervisor'].map({'Yes':1,'No':0,'Some of them':0.5})
# data_test['mental_health_interview']=data_test['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
data_test['phys_health_interview']=data_test['phys_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
# data['mental_vs_physical']=data['mental_vs_physical'].map({'Yes':1,'No':0,"Don't know":0.5})
# data_test['obs_consequence']=data_test['obs_consequence'].map({'Yes':1,'No':0})

X_test=data_test.loc[:, data_test.columns != 'treatment']
print(X_test)
y_pred = logreg.predict(X_test)

y_pred=np.reshape(y_pred,(259,1),order='C')
print(y_pred.shape)

# #Predicts only for one value
# one_array=np.array([0,0])
# one_array=np.reshape(one_array,(-1,1),order='C').T
# one_value=logreg.predict(np.array([one_array[0][0]]).reshape((-1,1),order='C'))
# print("One value prediction is:",one_value)


my_df = pd.DataFrame(y_pred)
my_df.to_csv('output_test_final.csv', index=False)


# save the model to disk
filename = 'Logic_model_final.pkl'
joblib.dump(logreg, filename)
# my_df['treatment']=my_df['treatment'].map({1:'Yes',0:'No'})
# data_test['treatment']=my_df['treatment']
#
# data

loaded_model = joblib.load(filename)
result = loaded_model.predict(X_test)
print(result)
