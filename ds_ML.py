# Trivago session analysis by IWONA SMITH 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

#importing data
df = pd.read_csv('session_actions.csv')

# data summary
df.info()
df.describe()
df.isnull().sum()



##### Action Ref missing values
df['action_reference'].isnull().sum()
# fill in with most common value
df['action_reference'] = df['action_reference'].fillna(1)

df['action_reference']=df['action_reference'].astype(int)
df['action_id'] = df['action_id'].astype(int)
df['step'] = df['step'].astype(int)


category = list(df['action_id'].unique()) # 102 unique
context= list(df['action_reference'].unique()) # 89836 unique
rank = list(df['step'].unique()) # 1957 unique


# exploring top action id
#•••••••••••••••••••••••••••
df.action_id.value_counts('3').head(10)
sum(df.action_id.value_counts('3').head(10))

print(df["action_reference"][df['action_id'] == 2142].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 2113].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 2160].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 2111].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 2473].value_counts(10).head()) #### only 2 references
print(df["action_reference"][df['action_id'] == 8001].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 2292].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 2114].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 6001].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 2455].value_counts(10).head())
print(df["action_reference"][df['action_id'] == 2296].value_counts(10).head())


print(df["step"][df['action_id'] == 2142].value_counts(10).head())
print(df["step"][df['action_id'] == 2113].value_counts(10).head())
print(df["step"][df['action_id'] == 2160].value_counts(10).head())
print(df["step"][df['action_id'] == 2111].value_counts(10).head())
print(df["step"][df['action_id'] == 2473].value_counts(10).head()) 
print(df["step"][df['action_id'] == 8001].value_counts(10).head())
print(df["step"][df['action_id'] == 2292].value_counts(10).head())
print(df["step"][df['action_id'] == 2114].value_counts(10).head())
print(df["step"][df['action_id'] == 6001].value_counts(10).head())
print(df["step"][df['action_id'] == 2455].value_counts(10).head())



## exploring top step values
#•••••••••••••••••••••••••••
sum(df.step.value_counts('3').head())
df.step.value_counts('3').head()

pd.crosstab(df['action_id'],[df['step'] == 1])
pd.crosstab(df['action_id'],[df['step'] == 3])
pd.crosstab(df['action_id'],[df['step'] == 8])
pd.crosstab(df['action_id'],[df['step'] == 12])
pd.crosstab(df['action_id'],[df['step'] == 15])
pd.crosstab(df['action_id'],[df['step'] == 25])

## exploring top action_reference
#•••••••••••••••••••••••••••••••
sum(df.action_reference.value_counts('3').head())
df.action_reference.value_counts('3').head(10)

pd.crosstab(df['step'], [df['action_reference'] == 1])# 6-8 step
pd.crosstab(df['step'], [df['action_reference'] == 0])# 8-9 step
pd.crosstab(df['step'], [df['action_reference'] == 62])# 11-12 step
pd.crosstab(df['step'], [df['action_reference'] == 2])# 13-21 step
pd.crosstab(df['step'], [df['action_reference'] == 63])# 13-16 step



#visualisation

#how actions are distributed

plt.hist(df['action_id'])
plt.title("ActionId distribution")
plt.hist(df['action_reference'])
plt.title("ActionReference distribution")
plt.hist(df['step'])
plt.title("Step distribution")



sns.boxplot(df['action_reference'])
sns.boxplot(df['step'])
sns.boxplot(df['action_id'])

sns.distplot(df['action_id'], color='#FD5C30')
plt.xlabel('action_id')
sns.despine()

sns.distplot(df['action_reference'][0:64], color='#DD5C64')
plt.xlabel('action_reference')
sns.despine()

sns.distplot(df['step'], color='#FD5C64')
plt.xlabel('step')
sns.despine()
sns.distplot(df['step'][0:60], color='#FD5C64')
plt.xlabel('step')
sns.despine()




## whats the most common action_id ?

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.stripplot(x="action_id", y='step'  , data=df)
plt.xticks(rotation=40)

print(df["step"][df['action_id'] == 2142])



df['action_reference'].value_counts(10).head(100)

plt.hist(df['step'].value_counts())

plt.scatter('action_reference', 'step', data=df)

plt.hist(df['action_id'].value_counts())
## whats the most common step ?
sum(df['step'].value_counts(10).head(100))
df['step'].value_counts(10).head(100)
plt.hist(df['step'].value_counts())


sns.distplot(df['step'].value_counts(), color='#FD5C64')
plt.xlabel('Step')
sns.despine()




