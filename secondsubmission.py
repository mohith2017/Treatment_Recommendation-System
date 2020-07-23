#Second submission
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data=pd.read_csv('trainms.csv')
data_test=pd.read_csv('testms.csv')
data_output_test=pd.read_csv('samplems.csv')

data_test.head(10)

data['treatment']=data['treatment'].map({'Yes':1,'No':0})
data['mental_health_interview']=data['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
data_test['mental_health_interview']=data_test['mental_health_interview'].map({'Yes':1,'No':0,'Maybe':0.5})
data_output_test['treatment']=data_output_test['treatment'].map({'Yes':1,'No':0})
data['self_employed']=data['self_employed'].map({'Yes':1,'No':0})
data['family_history']=data['family_history'].map({'Yes':1,'No':0})
data['work_interfere']=data['work_interfere'].map({'Often':1,'Sometimes':0.8,'Never':0,'Rarely':0.4})
data['work_interfere']=data['work_interfere'].dropna(how='any',axis=0)
data['remote_work']=data['remote_work'].map({'Yes':1,'No':0})
data['remote_work']=data['remote_work'].map({'Yes':1,'No':0})
data['tech_company']=data['tech_company'].map({'Yes':1,'No':0})
print(data['work_interfere'])
print(data['treatment'])

count_no_sub = len(data[data['treatment']==0])
count_sub = len(data[data['treatment']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no treatment is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of treatment", pct_of_sub*100)

print(data['treatment'].value_counts())

print(data.groupby('self_employed').mean())

pd.crosstab(data.self_employed,data.treatment).plot(kind='bar')
plt.title('Whether treatment is required or not')
plt.xlabel('self_employed')
plt.ylabel('Frequency of requirement')
plt.show()

plt.bar(data['mental_health_interview'], data['treatment'])
plt.xlabel('Mental health interview taken?')
plt.ylabel('Treatment required?')
# plt.xticks(index, label, fontsize=5, rotation=30)
plt.title('Requirement of treatment')
plt.show()

data_final=data[['mental_health_interview','treatment']]
# ,'treatment','self_employed','family_history','work_interfere','remote_work','tech_company','benefits','care_options','wellness_program','seek_help','anonymity','leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','phys_health_interview','mental_vs_physical','obs_consequence']]

X = data_final.loc[:, data_final.columns != 'treatment']
y = data_final.loc[:, data_final.columns == 'treatment']

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
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


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train=data['mental_health_
logreg = LogisticRegression()
logreg.fit(X_train, np.ravel(y_train,order='C'))


y_pred = logreg.predict(X_test)



print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

data_test=data_test[['mental_health_interview']]
X_test=data_test.loc[:, data_test.columns != 'treatment']


y_pred = logreg.predict(X_test)
y_pred=np.reshape(y_pred,(259,1),order='C')


my_df = pd.DataFrame(y_pred)
my_df.to_csv('output_test_DS.csv', index=False)

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
# fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
# plt.show()
