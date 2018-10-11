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
df = pd.read_csv('session_data.csv')
df.info()
df.describe()
df.isnull().sum()


df = df.drop([ 'session_date_id', 'tracking_id', 'traffic_type' ], axis = 1)

# row values to be droped as irrelevant
df = df.drop(df[df["entry_page"] == 2106].index)
df = df.drop(df[df["entry_page"] == 2116].index)

df = df.drop(df[df["bookings"] == 2].index)
df = df.drop(df[df["bookings"] == 3].index)
df = df.drop(df[df["bookings"] == 4].index)


# Date 
df['ymd']= df['ymd'].astype(str)
df['ymd'] = df['ymd'].apply(lambda x:datetime.datetime.strptime(x, '%Y%m%d'))
df['ymd'] = df['ymd'].astype(str)

# split data to test and control
df_test = df[df['test_on'] == 1]
df_test = df_test.drop(['test_on'], axis = 1)
df_control = df[df['test_on'] == 0]
df_control = df_control.drop(['test_on'], axis = 1)



#### KPI clickouts
# clickouts 
# ••••••••••••••••••
sns.lmplot(x='clickouts', y='image_ctp', fit_reg=False, data=df_test, hue = 'entry_page')
plt.title("Test")
sns.lmplot(x='clickouts', y='image_ctp', fit_reg=False, data=df_control, hue = 'entry_page')
plt.title("Control")


#  clickouts depending on search_type 
# ••••••••••••••••••

sns.lmplot(x='clickouts', y='search_type', fit_reg=False, data=df_test, hue = 'entry_page')
plt.title("Test")
sns.lmplot(x='clickouts', y='search_type',  fit_reg=False, data=df_control, hue = 'entry_page')
plt.title("Control")

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='clickouts',hue='search_type', data=df_control, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0,10])
plt.ylim([0,55000])
plt.title("Control")

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='clickouts',hue='search_type', data=df_test, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0,10])
plt.ylim([0,55000])
plt.title("Test")


# clickouts by day
# ••••••••••••••••••
ts = df_control.groupby("ymd")[ "clickouts"].sum()
ts.plot(figsize=(10,6), linewidth=3, fontsize=10, color="orange")
zs = df_test.groupby("ymd")[ "clickouts"].sum()
zs.plot(figsize=(10,6), linewidth=3, fontsize=10, color="r" )



fig, ax = plt.subplots()
sns.kdeplot(ts, ax=ax, color="orange")
sns.kdeplot(zs, ax=ax , color="r")

fig, ax = plt.subplots()
sns.kdeplot(df_control["clickouts"],shade=True, color="orange", ax=ax)
sns.kdeplot(df_test["clickouts"], shade=True, color="r", ax=ax)
sns.plt.xlim(-0.5,10)

fig, ax = plt.subplots()
sns.kdeplot(df_control["clickouts"],shade=True, color="orange", ax=ax)
sns.kdeplot(df_test["clickouts"], shade=True, color="r", ax=ax)
sns.plt.xlim(-0.5,)


# clickouts per day 
sns.stripplot(x="ymd", y="clickouts", data=df_test)
plt.xticks(rotation=90)
plt.title("Test")
sns.stripplot(x="ymd", y="clickouts", data=df_control)
plt.xticks(rotation=90)
plt.title("Control")
# •••••••••••••••••••••••••••••••
# •••••••••••••••••••••••••••••••




# KPI session_duration
# •••••••••••••••••••••••••••••••
sns.lmplot(x='session_duration', y='clickouts', fit_reg=False, data=df_test, hue = 'device')
sns.lmplot(x='session_duration', y='clickouts',  fit_reg=False, data=df_control, hue = 'device')

sns.lmplot(x='session_duration', y='clickouts', fit_reg=False, data=df_test, hue = 'entry_page')
sns.lmplot(x='session_duration', y='clickouts',  fit_reg=False, data=df_control, hue = 'entry_page')

# •••••••••••••••••••••••••••••••
fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='session_duration',hue='search_type', data=df_control, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0.5,50])
plt.ylim([0,3000])
plt.title("Control")

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='session_duration',hue='search_type', data=df_test, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0.5,50])
plt.ylim([0,3000])
plt.title("Test")
# •••••••••••••••••••••••••••••••
# •••••••••••••••••••••••••••••••



#KPI content engagement
# •••••••••••••••••••••••••••••••
fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='image_ctp',hue='device', data=df_control, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0.5,10.5])
plt.ylim([0,5200])
plt.title("Control")

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='image_ctp',hue='device', data=df_test, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0.5,10.5])
plt.ylim([0,5200])
plt.title("Test")


# •••••••••••••••••••••••••••••••
fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='image_ctp',hue='is_repeater', data=df_control, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0.5,10.5])
plt.ylim([0,5000])
plt.title("Control")

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='image_ctp',hue='is_repeater', data=df_test, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0.5,10.5])
plt.ylim([0,5000])
plt.title("Test")

# ••••••••••••••••••
# ••••••••••••••••••







############# SEASONALITY
import statsmodels.api as sm
# multiplicative
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()


# Additive model
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()

##################


fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='entry_page',hue='device', data=df_test, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.legend(country)

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='entry_page',hue='device', data=df_control, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.legend(country)



# •••••••••••••••••••••••••••••••
fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='agent_id',hue='search_type', data=df_control, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0.5,10])
plt.ylim([0,8000])

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='agent_id',hue='search_type', data=df_test, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.xlim([0.5,10])
plt.ylim([0,8000])

# ••••••••••••••••••

#••••••••••••••• session duration  test
sns.stripplot(x="ymd", y="session_duration", data=df_test)
plt.xticks(rotation=90)

sns.stripplot(x="ymd", y="session_duration", data=df_control)
plt.xticks(rotation=90)

#•••••••••••••••



# KPI user by top country 

#••••••••••••••••

country = {'Turkey':1 , 'United Kingdom': 2, 'France' :3 }
            
df_test['country_val'] = df_test["countryname"].map(country)
df_control['country_val'] = df_control["countryname"].map(country)


fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='is_repeater',hue='country_val', data=df_test, palette="husl", ax=axis1)
plt.legend(country)

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='is_repeater',hue='country_val', data=df_control,  palette="husl", ax=axis1)
plt.legend(country)

#••••••••••••••••••
