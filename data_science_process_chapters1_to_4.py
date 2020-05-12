# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:27:28 2020

@author: BetaCosine
"""

'''
Once the data science environment is set up
the next step is to connect to the data base store
This program is intended to walk through each step
in the data science development process

In the next program we will create code based on code here
but for production purposes

'''
#First, import the libraries you need

import sqlite3 
import pandas as pd
import math 
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve 
import matplotlib.pyplot as plt


#Second, connect to your data

db_path = 'C:/sqlite/'
db_name = 'employ.db'

con = sqlite3.connect(db_path+db_name)

#Third, build your dataframe using Approach 1 in the book

sql = """
SELECT a.SEP, a.AGELVL, a.PATCO, a.WORKSCH, a.SALARY, a.LOS, b.SEPT, 
c.AGELVLT, d.PATCOT, e.WORKSCHT 
FROM
SEPDATA_FY2016 a
LEFT JOIN
DTsep b
ON a.sep = b.sep
LEFT JOIN
DTagelvl c
ON a.agelvl = c.agelvl
LEFT JOIN
DTpatco d
ON a.patco = d.patco
LEFT JOIN
DTwrksch e
ON a.worksch = e.worksch
"""

df = pd.read_sql(sql, con)

#Fourth, define the target variables

#first we build a function to recode 
#one of the columns, known as a Series in pandas

def binary_target(series):
    if series == "Quit":
        return 1
    else:
        return 0

#then we apply the function to the series with the apply method

df['QuitGroup'] = df['SEPT'].apply(binary_target)

#drop the old target columns so as not to confuse ourselves later

df.drop(['SEP','SEPT'], axis=1, inplace=True)

#create the second dataframe with numerical target

#first we need to split our data set to only include those who quit

df_target2 = df.loc[df['QuitGroup'] == 1]

#next we drop the first target column
#the LOS column will be our numeric target variable for the second model

df_target2.drop('QuitGroup', axis=1, inplace = True)

'''
At this point we typically want to perform some exploratory data analysis or 
EDA. The example code to do this is covered under the Understanding Your 
Data (Exploratory Data Analysis [EDA]) section of chapter 2. I have created
a separate program just for EDA, which is a common practice in development.
Because we have performed the EDA we already know how we want to 
feature engineer our variables, so here we include our feature engineering 
code.
'''

#before we start feature engineering, let's clean up the dataframe

drop_list_1 = ['AGELVL', 'PATCO', 'WORKSCH']
df.drop(drop_list_1, inplace=True, axis = 1)
df_target2.drop(drop_list_1, inplace=True, axis=1)


#Fifth, feature engineering

#categorical variables

def work_sch(series):
    if series == 'F-Full-time Nonseasonal':
        return 'wk_sch_F'
    else:
        return 'wk_sch_other'

df['work_sch_groups'] = df['WORKSCHT'].apply(work_sch)
df_target2['work_sch_groups'] = df_target2['WORKSCHT'].apply(work_sch)
df.drop('WORKSCHT', inplace=True, axis=1)
df_target2.drop('WORKSCHT', inplace=True, axis=1)

#dummy code remaining categorical variables
df = pd.get_dummies(df, drop_first = True)
df_target2 = pd.get_dummies(df_target2, drop_first = True)

#feature engineering numeric variables

#fill in missing numeric variables with the mean
df['SALARY'].fillna(df['SALARY'].mean(), inplace=True)
df['LOS'].fillna(df['LOS'].mean(), inplace=True)

df_target2['SALARY'].fillna(df_target2['SALARY'].mean(), inplace=True)
df_target2['LOS'].fillna(df_target2['LOS'].mean(), inplace=True)

#check to ensure fill worked with 
#nulls = df.isnull().sum().to_frame()

#create new numeric features with transformations

df['salary_root'] = df['SALARY'].apply(math.sqrt)
df['salary_log'] = df['SALARY'].apply(math.log)
df['salary_inv'] = 1/df['SALARY']

df['los_root'] = df['LOS'].apply(math.sqrt)
'''
note that the below code will not work because we cannot take the log 
of 0, and there are some LOS values that are 0
i leave it here just for your own edification

df['los_log'] = df['LOS'].apply(math.log)

'''
df['los_inv'] = 1/df['LOS']

df_target2['salary_root'] = df_target2['SALARY'].apply(math.sqrt)
df_target2['salary_log'] = df_target2['SALARY'].apply(math.log)
df_target2['salary_inv'] = 1/df_target2['SALARY']

'''
note that for each of the numeric transformations we run the risk of 
infinite values or the creation of nan's if we do not get errors
for example, in the inverse calculation of 1/0 pandas will return a
numpy value of inf or -inf, meaning an infinate value
therefore, we run the following code to be sure
'''

df = df.replace([np.inf, -np.inf, np.nan], 0)
df_target2= df_target2.replace([np.inf, -np.inf, np.nan], 0)

#Next is to perform any additional feature selection
#the code is part of the exploratory_data_analysis.py program
#based on those efforts we identify 
# variables to drop for the first modeling scenario
drop_list = ['SALARY', 'salary_inv', 'salary_root', 'los_root']
df.drop(drop_list, axis=1, inplace=True)

#variables to drop for the second modeling scenario

drop_list2 = ['SALARY', 'salary_inv', 'salary_root']
df_target2.drop(drop_list2, axis=1, inplace=True)

#next we separate our target variables from the features

#Create separate series of just target variable, Y

Y = df['QuitGroup']
Y2 = df_target2['LOS']

X = df.drop('QuitGroup', axis=1)
X2 = df_target2.drop('LOS', axis=1)

#next we normalize the features

X = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X)
X = min_max_scaler.transform(X)

X2 = X2.values #returns a numpy array
min_max_scaler2 = preprocessing.MinMaxScaler()
min_max_scaler2.fit(X2)
X2 = min_max_scaler2.transform(X2)

#in order to save our scalers

from joblib import dump, load
#choose path to save to
path = 'C:/Users/BetaCosine/Google Drive/BetaCosine/Ebooks/'

#save the scalers
dump(min_max_scaler, path+'scaler_binary.joblib')
dump(min_max_scaler2, path+'scaler_numerical.joblib')

#finally, save the features for each model to a csv for production

features_binary = pd.DataFrame((df.drop('QuitGroup', axis=1)).columns)
features_binary.to_csv(path+'features_binary.csv')
features_numerical = pd.DataFrame((df_target2.drop('LOS', axis=1)).columns)
features_numerical.to_csv(path+'features_numeric.csv')

#The next step is to split each set of data into train and test sets

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .33, 
                                                    random_state=42)

x2_train, x2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size = .33, 
                                                    random_state=42)

#test models for first, categorical scenario

lr = LogisticRegression()
rf = RandomForestClassifier()
nb = BernoulliNB()

lr.fit(x_train, y_train)
rf.fit(x_train, y_train)
nb.fit(x_train, y_train)

pred_lr = lr.predict(x_test)
pred_rf = rf.predict(x_test)
pred_nb = nb.predict(x_test)

conf_mat_lr = confusion_matrix(y_test, pred_lr)
conf_mat_rf = confusion_matrix(y_test, pred_rf)
conf_mat_nb = confusion_matrix(y_test, pred_nb)

tn_lr, fp_lr, fn_lr, tp_lr = conf_mat_lr.ravel()
tn_rf, fp_rf, fn_rf, tp_rf = conf_mat_rf.ravel()
tn_nb, fp_nb, fn_nb, tp_nb = conf_mat_lr.ravel()

ppv_lr = tp_lr/(tp_lr+fp_lr)
ppv_rf = tp_rf/(tp_rf+fp_rf)
ppv_nb = tp_nb/(tp_nb+fp_nb)

#plot ROC curve for logistic regression
lr_prob = lr.predict_proba(x_test).reshape(-1,2)
fpr, tpr, _ = roc_curve(y_test, lr_prob[:,1])

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#test models for the second scenario, numerical prediction

linr = LinearRegression()
rfreg = RandomForestRegressor()
rr = Ridge()

linr.fit(x2_train, y2_train)
rfreg.fit(x2_train, y2_train)
rr.fit(x2_train, y2_train)

linr_r2 = linr.score(x2_test, y2_test)
rfreg_r2 = rfreg.score(x2_test, y2_test)
rr_r2 = rr.score(x2_test, y2_test)

#run a loop to print each r-squared value for each model
#to the console

r2_list = [linr_r2,rfreg_r2,rr_r2]
for i in r2_list:
    print(i)
    
#example ensemble for our binary classification models
#here we use the average probability from the 3 models 
#to decide whether an observation (row) should be labeled as a "1"

#create arrays of predicted probabilities for each model
lr_prob = lr.predict_proba(x_test).reshape(-1,2)
rf_prob = rf.predict_proba(x_test).reshape(-1,2)
nb_prob=  nb.predict_proba(x_test).reshape(-1,2)

#combine the probabilities for class 1 into single array
combined = np.concatenate((lr_prob[:,1].reshape(-1,1),
                           rf_prob[:,1].reshape(-1,1),
                           nb_prob[:,1].reshape(-1,1)),axis=1)

#find the average prbability and put into pandas dataframe   
average = pd.DataFrame(combined.mean(axis=1))

#function for deciding whether average probability is a 1 or 0
def ens_dec(series):
    if series <= .5:
        return 0
    if series > .5:
        return 1

#apply function to the dataframe titled "average"
ens_score = average[0].apply(ens_dec)

#compute ppv to see how our ensemble performs 
conf_mat_ens = confusion_matrix(y_test, ens_score)
tn_nb, fp_nb, fn_nb, tp_nb = conf_mat_ens.ravel()
ppv_ens = tp_lr/(tp_lr+fp_lr)

#saving models

path = 'C:/Users/BetaCosine/Google Drive/BetaCosine/Ebooks/'
dump(lr, path+'quitters_model_lr.joblib') 
dump(linr,path+'los_model_linr.joblib')

