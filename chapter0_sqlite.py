# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sqlite3 
import csv
import os
from pathlib import Path
import pandas as pd

path = 'C:/your_data_path/'
db_path = 'C:/sqlite/'
db_name = 'employ.db'

def return_data(path, file_type='.txt'):
    data_list = []
    for file in os.listdir(path):
        if file.lower().endswith(file_type):
            data_list.append(file)
    return data_list


def db_connect(db_path, db_name):
    con = sqlite3.connect(db_path+db_name)
    return con


def load_sqlite(path, text_file, conn):
    df = pandas.read_csv(csvfile)
    df.to_sql(table_name, conn, if_exists='append', index=False)

#get list of text data files
data_list = return_data(path)

#connect to database
con = sqlite3.connect(db_path+db_name)
cur = con.cursor()

#load data from text files to database
for file in data_list:
    df = pd.read_csv(path+file)
    table_name = Path(file).stem
    df.to_sql(table_name,con, if_exists='append', index=False)


"""
Playing with data
"""

sql = """
SELECT * from SEPDATA_FY2016 LIMIT 100
"""

df_test = pd.read_sql(sql, con)

"""
Joining Data 
"""

sql = """
SELECT a.*, b.SEPT, c.AGELVLT, d.PATCOT, e.WORKSCHT 
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

"""
Data wrangling
"""

def binary_target(series):
    if series == "Quit":
        return 1
    else:
        return 0

df['QuitGroup'] = df['SEPT'].apply(binary_target)
df.drop(['SEP','SEPT'], axis=1, inplace=True)

df.columns

df_target2 = df.loc[df['QuitGroup'] == 1]

df_target2.drop('QuitGroup', inplace = True)

df_target2.dtypes

#checking data

nulls = df.isnull().sum().to_frame()
nulls['proportion'] = nulls/df.count().max()

df.nunique()

#descriptive statistics, numeric

describe = df[['SALARY', 'LOS']].describe()


#descriptive statistics, categorical

freq_wksch = df['WORKSCH'].value_counts()/df.count().max()

#visualizations, histograms

df[['SALARY', 'LOS']].hist()

#visualizations, bar charts

freq_wksch.plot.bar()

'''
feature engineering

'''

#numeric feature engineering

import math

df['salary_root'] = df['SALARY'].apply(math.sqrt)
df['salary_log'] = df['SALARY'].apply(math.log)
df['salary_inv'] = 1/df['SALARY']

df[['SALARY', 'salary_log']].hist()

#imputing missing values with mean

from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(df[['SALARY', 'salary_root', 'salary_log', 'salary_inv']])
df[['SALARY', 'salary_root', 'salary_log', 'salary_inv']] = imputer.transform(
        df[['SALARY', 'salary_root', 'salary_log', 'salary_inv']])

#categorical feature engineering

def work_sch(series):
    if series == 'F':
        return 'wk_sch_F'
    else:
        return 'wk_sch_other'

df['work_sch_groups'] = df['WORKSCH'].apply(work_sch)

#drop the old categorical column

df.drop('WORKSCH', inplace = True)

#once complete removing the unwanted columns
#one hot encoding



#connect to remote database


import mysql.connector

remote_db = mysql.connector.connect(
  host="db780643100.hosting-data.io",
  port=3306,
  user="db780643100",
  passwd=
)
