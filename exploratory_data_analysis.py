# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:52:44 2020

@author: BetaCosine
"""
'''
the code herein assumes you have already set up your base dataframes
using the data_science_process_chapters1_to_4.py code up to the EDA
section.  
'''
import math

#checking the data types in the dataframe

df.dtypes
df_target2.dtypes

#checking the number of unique values for each column
df.nunique()
df_target2.nunique()

#checking for missing values
#first we make a missing dataframe, then create a proportion column
#this allows us to compare the overall impact of missing values

nulls = df.isnull().sum().to_frame()
nulls['proportion'] = nulls/df.count().max()

nulls2 = df_target2.isnull().sum().to_frame()
nulls2['proportion'] = nulls2/df_target2.count().max()

#descriptive statistics for numeric variables

describe = df[['SALARY', 'LOS']].describe()
describe2 = df_target2[['SALARY', 'LOS']].describe()

describe_all = df.describe()
describe_all2 = df_target2.describe()

#descriptive statistics for categorical variables; frequencies
#create frequency distribution dataframes to look at frequency for each category
#here is one example

dataframe = df #you can change this to be df_target2 for the other set

freq_agelvl = dataframe['AGELVL'].value_counts()
freq_patco = dataframe['PATCO'].value_counts()
freq_wksch = dataframe['WORKSCH'].value_counts()
freq_agelvlt = dataframe['AGELVLT'].value_counts()
freq_patcot = dataframe['PATCOT'].value_counts()
freq_wkscht = dataframe['WORKSCHT'].value_counts()

#even better is to look at proportions in each category
#here is one example

freq_wksch = df['WORKSCH'].value_counts()/df.count().max()

#Visualizations of the data

# Histograms for numeric variables

df[['SALARY', 'LOS']].hist()

# Bar charts for categorical variables
#notice that we use the frequency dataframe we made above

freq_wksch.plot.bar()

'''
Now that we have learned more about our data we need to make decisions
regarding how we will feature engineer each in order to get closer to 
our final dataframe that will be ready for machine learning.

Importantly, we need to copy the code below into our overall process
so that we don't forget to include this feature engineering code
in our final production code. 
'''

#feature engineering categorical variables

def work_sch(series):
    if series == 'F':
        return 'wk_sch_F'
    else:
        return 'wk_sch_other'

df['work_sch_groups'] = df['WORKSCH'].apply(work_sch)

#example to fill in missing categorical values with the mode

df['WORKSCHT'].fillna(df['WORKSCHT'].mode()[0], inplace=True)

df_target2['WORKSCHT'].fillna(df_target2['WORKSCHT'].mode()[0], inplace=True)

#feature engineering numeric variables
df['salary_root'] = df['SALARY'].apply(math.sqrt)
df['salary_log'] = df['SALARY'].apply(math.log)
df['salary_inv'] = 1/df['SALARY']

df['los_root'] = df['LOS'].apply(math.sqrt)
df['los_log'] = df['LOS'].apply(math.log) #this will give an error
df['los_inv'] = 1/df['LOS']

#compare the histograms

df[['SALARY', 'salary_log']].hist()

#correlations

import seaborn as sns

sns.heatmap(df.corr())
sns.heatmap(df[['QuitGroup', 'LOS', 'SALARY', 'salary_root', 
                'salary_log', 'salary_inv', 'los_root', 'los_inv']].corr(),
    annot = True)
