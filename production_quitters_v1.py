# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:23:57 2020

@author: BetaCosine
"""
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
from joblib import dump, load

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve 
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=Warning)


'''
START OF PREPROCESSING CODE, TO INCLUDE:
    
    - CONNECT TO DATA SOURCE(S)
    - PULL DATA INTO MEMORY WITHOUT TARGET CODES
    - WRANGLE AND FEATURE ENGINEER DATA

'''
#set the project path where your scalers, models, and features are saved
path = 'C:/Users/BetaCosine/Google Drive/BetaCosine/Ebooks/'

#Second, connect to your data

db_path = 'C:/sqlite/'
db_name = 'employ.db'

con = sqlite3.connect(db_path+db_name)

#Third, build your dataframe using Approach 1 in the book
#note you may need to add a WHERE clause (not shown here) to your data pull to
#ensure you are only pulling the most recent data in the database

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

drop_list_1 = ['AGELVL', 'PATCO', 'WORKSCH']
df.drop(drop_list_1, inplace=True, axis = 1)


#Feature engineering

#categorical variables

def work_sch(series):
    if series == 'F-Full-time Nonseasonal':
        return 'wk_sch_F'
    else:
        return 'wk_sch_other'

df['work_sch_groups'] = df['WORKSCHT'].apply(work_sch)
df.drop('WORKSCHT', inplace=True, axis=1)

#dummy code remaining categorical variables
df = pd.get_dummies(df, drop_first = True)

#feature engineering numeric variables

#fill in missing numeric variables with the mean
df['SALARY'].fillna(df['SALARY'].mean(), inplace=True)
df['LOS'].fillna(df['LOS'].mean(), inplace=True)

#create new numeric features with transformations

df['salary_root'] = df['SALARY'].apply(math.sqrt)
df['salary_log'] = df['SALARY'].apply(math.log)
df['salary_inv'] = 1/df['SALARY']

df['los_root'] = df['LOS'].apply(math.sqrt)

df['los_inv'] = 1/df['LOS']

df = df.replace([np.inf, -np.inf, np.nan], 0)

#split into two dataframes to properly normalize

features_binary = pd.read_csv(path+'features_binary.csv')
features_numerical = pd.read_csv(path+'features_numeric.csv')
df_binary = df[features_binary['0'].unique()]
df_numerical = df[features_numerical['0'].unique()]

##load our scalars to complete normalization 

min_max_scaler = load(path+'scaler_binary.joblib')
min_max_scaler2 = load(path+'scaler_numerical.joblib')

X = df_binary.values
X = min_max_scaler.transform(X)

X2 = df_numerical.values
X2 = min_max_scaler2.transform(X2)

#load the models

lr = load(path+'quitters_model_lr.joblib') 
linr = load(path+'los_model_linr.joblib')

#score the models and add the predictions as columns to their 
#dataframes

df_binary['quit_class'] = lr.predict(X)
df_numerical['los_pred'] = linr.predict(X2)

#add the los_pred predicted values to one dataframe

df_merged = pd.merge(df_binary, df_numerical['los_pred'], 
                     left_index=True, right_index=True)

#build priority list to complete solution

#create column of difference between predicted LOS and actual LOS
df_merged['los_diff'] = df_merged['los_pred'] - df_merged['LOS']

#subset dataframe to include just those predicted to quit
df_quit = df_merged.loc[df_merged['quit_class'] == 1]

#create dataframe with the most at risk for quitting the soonest
at_risk_1000 = df_quit.sort_values('los_diff', ascending=True).head(1000)

#save list to a .csv for your boss!
at_risk_1000.to_csv(save_path+'at_risk_1000.csv')
