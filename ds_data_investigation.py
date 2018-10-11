
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


columns = list(df.columns.values)
print(columns)


df = df.drop([ 'session_date_id', 'tracking_id', 'traffic_type' , 'test_on'], axis = 1)


# Date 
df['ymd']= df['ymd'].astype(str)
df['ymd'] = df['ymd'].apply(lambda x:datetime.datetime.strptime(x, '%Y%m%d'))
df['ymd'] = df['ymd'].astype(str)

# session analysis
df.info()
df.describe()
df.isnull().sum()


# time series
#•••••••••••••
df["ymd"] = df["ymd"].astype("int")
df.ymd.value_counts().sort_index().plot(kind='bar',stacked=True)

# by day of the week 
df["ymd"] = df["ymd"].astype("datetime64")
df.groupby(df["ymd"].dt.weekday_name)['entry_page'].sum().sort_index().plot(kind='bar')
df.groupby(df["ymd"].dt.weekday_name).count().plot(kind="bar")



# user clickouts per day
#•••••••••••••••••••••••••
ts = df.groupby("ymd")[ "clickouts"].sum()
ts.plot(figsize=(10,6), linewidth=3, fontsize=10)
plt.title('Total clickouts per day')
plt.xlabel('TIME')
plt.ylabel('clickouts')
df.groupby("ymd")["clickouts"].sum().plot.bar()

pd.crosstab(df['ymd'], df['clickouts'])

# clickouts and search type
df.clickouts.value_counts().sort_index().plot(kind='bar',stacked=True)

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(df.clickouts, hue='search_type', data=df, palette="husl",order=df.clickouts.value_counts().iloc[:5].index, ax=axis1)

# clickouts vs. image_ctp by returning user
sns.lmplot(x='clickouts', y='image_ctp', fit_reg=False, data=df, hue = 'is_repeater')
sns.plt.ylim(-1,1000)
sns.plt.xlim(-1,21)

plt.scatter(df.clickouts,df.image_ctp)

# clickouts v. bookings
fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot( x='clickouts',hue='bookings', data=df, palette="husl",order=df.clickouts.value_counts().iloc[:10].index, ax=axis1)



# user by country 
#••••••••••••••••••••••••••••
df["countryname"].value_counts('20').head(10)
df["countryname"].isnull().sum()
#missing 171 values in coutry name
# top 3 countries
country = {'Turkey':1 , 'United Kingdom': 2, 'France' :3 }            
df['country_val'] = df["countryname"].map(country)

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='ymd',hue='country_val', data=df, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.legend(country)


x=df.groupby(df['countryname']).count()
x=x.sort_values(by='bookings',ascending=False)
x=x.iloc[:10].reset_index()

plt.figure(figsize=(10,6))
ax = sns.barplot(x.countryname, x.bookings, alpha=0.8)
plt.title("Session data - country split")
plt.ylabel('# counts', fontsize=12)
plt.xlabel('', fontsize=12)
plt.show()



#  KPI users country of orgin
df['Turkey']=df['countryname'].apply(lambda x: 1 if x == 'Turkey' else 0)
df['UK']=df['countryname'].apply(lambda x: 1 if x == 'United Kingdom' else 0)
df['France']=df['countryname'].apply(lambda x: 1 if x == 'France' else 0)
df['Germany']=df['countryname'].apply(lambda x: 1 if x == 'Germany' else 0)
df['Italy']=df['countryname'].apply(lambda x: 1 if x == 'Italy' else 0)

df2 =df[['ymd', 'Turkey','UK' , 'France', 'Germany', 'Italy']]
df2.groupby('ymd')['Turkey','UK' ,'France', 'Germany', 'Italy'].sum().plot( )


# KPI device use over duration of session

df['mobile']=df['device'].apply(lambda x: 1 if x == 'mobile' else 0)
df['desktop']=df['device'].apply(lambda x: 1 if x == 'desktop' else 0)
df['tablet']=df['device'].apply(lambda x: 1 if x == 'tablet' else 0)
df3 = df[['ymd', 'mobile', 'tablet', 'desktop']]
df3.groupby('ymd')['mobile', 'tablet', 'desktop'].sum().plot( )
plt.xticks(rotation=90)




#entry_page
#••••••••••••••••••••••• 
df["entry_page"].value_counts()
sns.lmplot(x='clickouts', y='ctp_total', fit_reg=False, data=df, hue = 'entry_page')


# KPI identifying the entry page variability 
df['page 2113']=df['entry_page'].apply(lambda x: 1 if x == 2113 else 0)
df['page 2111']=df['entry_page'].apply(lambda x: 1 if x == 2111 else 0)
df['page 2114']=df['entry_page'].apply(lambda x: 1 if x == 2114 else 0)
df4 = df[['ymd','page 2113', 'page 2111', 'page 2114']]
df4.groupby('ymd')['page 2113', 'page 2111', 'page 2114'].sum().plot( )
plt.xticks(rotation=90)
plt.title('Users volume  by entry page ' )


# KPI users by search type
df['type4']=df['search_type'].apply(lambda x: 1 if x == 4 else 0)
df['type1']=df['search_type'].apply(lambda x: 1 if x == 1 else 0)
df['type2']=df['search_type'].apply(lambda x: 1 if x == 2 else 0)
df['type3']=df['search_type'].apply(lambda x: 1 if x == 3 else 0)
df4 = df[['ymd','type1', 'type2', 'type3', 'type4' ]]
df4.groupby('ymd')['type1', 'type2', 'type3', 'type4' ].sum().plot( )
plt.xticks(rotation=90)
plt.title('Users by search type' )


# CTP content
#•••••••••••••••
# only 33% of users engages with the CTP content
df["ctp_total"] = pd.Series([df['image_ctp'] + df['info_ctp']+df['review_ctp']]).sum()
df.ctp_total.value_counts()
df.info_ctp.value_counts()

z = df.groupby('ymd')[ "ctp_total",'image_ctp','review_ctp', 'info_ctp'].sum()
z.plot(figsize=(10,6), linewidth=3, fontsize=10)
plt.legend( loc=1)
plt.xlable('ctp')

df8 = df[['image_ctp' ,'ymd']]
df8 = df8[~(df8 == 0).any(axis=1)]
df8.groupby('ymd')[ 'image_ctp'].sum().plot()


df.image_ctp.value_counts()
df.review_ctp.value_counts()
df.info_ctp.value_counts()
ys = df.groupby("ymd")[ "ctp_total"].sum()
ys.plot(figsize=(10,6), linewidth=3, fontsize=10)
plt.title('Total content per day')
plt.xlabel('TIME')
plt.ylabel('# counts')


fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='ymd',hue='ctp_total', data=df, palette="husl", ax=axis1)
plt.xticks(rotation=90)
plt.legend()



# device
#•••••••••••••••••
# sessions device split by user and date
fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot(x='ymd',hue='device', data=df, palette="husl", ax=axis1)
plt.xticks(rotation=90)




# session_duration
#••••••••••••••••••••••••
sns.stripplot(x="ymd", y="session_duration", data=df)
plt.xticks(rotation=90)

fig = plt.figure(); 
ax = fig.add_subplot(1, 1, 1)
ax.hist(df.session_duration, bins= 20,range=[100, 1000])  
plt.xlabel('session duration')
plt.ylabel('counts')
plt.show()

print(df['session_duration'].value_counts().tail(500))
df.session_duration.median()

plt.boxplot(df.session_duration)
sum(df.agent_id.value_counts('3').head(5))

df1 = (df.groupby("session_duration").filter(lambda x: len(x) > 200))

sns.boxplot(x=df['session_duration'])
sns.boxplot(x=df1['session_duration'])
sns.barplot(x="ymd", y='session_duration', data=df)
plt.xticks(rotation= 90)

ax = sns.boxplot(x="ymd", y='session_duration',  data=df,  palette="Set3")
plt.xticks(rotation= 90)

#session_duration and entry page
df1 = (df.groupby("session_duration").filter(lambda x: len(x) > 200))

fig, (axis1) = plt.subplots(1,1,sharex=True,figsize=(10,6))
sns.countplot( x='session_duration', hue = "entry_page", data=df, palette="husl", ax=axis1)
sns.plt.xlim(0, 20)
sns.plt.ylim(0, None)
















