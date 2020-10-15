# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:13:08 2020

@author: alex
"""

#Final Project - Data Analysis Practice II

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


features_train = pd.read_csv('C:/Users/alex/Documents/KyotoU/Data Analysis Practice II/Final_Project/dengue_features_train.csv')
features_test = pd.read_csv('C:/Users/alex/Documents/KyotoU/Data Analysis Practice II/Final_Project/dengue_features_test.csv')
labels_train = pd.read_csv('C:/Users/alex/Documents/KyotoU/Data Analysis Practice II/Final_Project/dengue_labels_train.csv')

#%%

y = labels_train['total_cases']
df = features_train
df = pd.concat([df,y], axis =1)

sj = df.iloc[:,0] == 'sj'
iq = df.iloc[:,0] == 'iq'

#%%

datearray = np.array(pd.to_datetime(df['week_start_date'])).astype(float)
df['week_start_date'] = pd.DataFrame(datearray).set_index(df.index)


for n, m in enumerate(list(df['city'].unique())):
    df['city']=df['city'].replace({m: n})       

df = df.dropna()

quicklook = []
for i in range(0,len(df.iloc[0,:])):
    quicklook.append(df.iloc[:,i].describe())

print(quicklook)


f = plt.figure(figsize=(5.5, 5))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(0,df.shape[1]), df.columns, fontsize=7, rotation=90)
plt.yticks(range(0,df.shape[1]), df.columns, fontsize=7)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=10)
#%%

fig,ax = plt.subplots(8,3,figsize = (12,20))

for i in range(0,8):
    for j in range(0,3):
        ax[i,j].scatter(df.iloc[:,i + 8*j][sj],df['total_cases'][sj],color = 'darkorange', label = 'San Juan',alpha =0.6)
        ax[i,j].scatter(df.iloc[:,i + 8*j][iq],df['total_cases'][iq], color = 'navy', label = 'Iquitos',alpha =0.6)
        ax[i,j].set_title(f'{df.columns[i + 8*j]} vs Total Cases')
        ax[i,j].set_xlabel(df.columns[i + 8*j])
        ax[i,j].set_ylabel('Total Cases')
        ax[i,j].legend()
        ax[i,j].locator_params(nbins=4)
        
fig.tight_layout()




#%%

from scipy import stats
averagetemp = pd.DataFrame({'city' : df['city'], 'average_temp_sj' : df['reanalysis_avg_temp_k']})

average_temp_sj = np.array(df['reanalysis_avg_temp_k'][sj])

average_temp_iq = df['reanalysis_avg_temp_k'][iq]
iq_mean = df['reanalysis_avg_temp_k'][iq].mean()

results = stats.ttest_1samp(average_temp_sj, iq_mean)

sig_val = 0.05

if results[1]/2 <= sig_val and df['reanalysis_avg_temp_k'][sj].mean() >= iq_mean:
    print()
    print(f'The p-value is smaller than the significance value and thus, the null hypothesis can be rejected, meaning San Juan can be deemed to have a higher average temperature than Iquitos and this may give us some insight into the number of Dengue fever cases across the two cities.')


#%%

#two sample t test for precipitation between the two cities
    
prepsj = df['precipitation_amt_mm'][sj]
prepiq = df['precipitation_amt_mm'][iq]

prepsj_mean = prepsj.mean()
prepiq_mean = prepiq.mean()

t,p    = stats.ttest_ind(prepsj, prepiq)

sig_val = 0.05
print()
print(f'p-value = {round(p,35)}')

if p <= sig_val:
    print()
    print(f'The p-value is smaller than the significance value and thus, the null hypothesis can be rejected, meaning San Juan and Iquitos are likely to have different amounts of weekly precipitation. This may give us some insight into the number of Dengue fever cases across the two cities.')


rprep  = (np.corrcoef(df['precipitation_amt_mm'][iq], df['total_cases'][iq])+np.corrcoef(df['precipitation_amt_mm'][sj], df['total_cases'][sj]))/2
print()
print(f'The correlation coefficient between precipitation and total cases is {round(rprep[0][1],3)}, so although the two cities have very different precipitation rates, there are certainly other forces at play here.')

#%%

from sklearn.cluster import KMeans

totalcases_normed = (df['total_cases']-df['total_cases'].mean())/df['total_cases'].max()
reanalysis_normed =  (df['reanalysis_tdtr_k']-df['reanalysis_tdtr_k'].mean())/df['reanalysis_tdtr_k'].max()
X = pd.concat([totalcases_normed,reanalysis_normed],axis = 1)
X = np.array(X)

kmeans = KMeans(n_clusters=2, random_state=1)
kmeans.fit(X)
labels = kmeans.labels_

fig,ax = plt.subplots(1,2,figsize = (12,5))
ax[0].scatter(X[:,1][np.array(labels,dtype = bool)],X[:,0][np.array(labels,dtype = bool)], c = 'indigo',label = 'San Juan', alpha = 0.7)
ax[0].scatter(X[:,1][np.invert(np.array(labels,dtype = bool))],X[:,0][np.invert(np.array(labels,dtype = bool))], c ='yellow', alpha = 0.7, label = 'Iquitos')
ax[0].set_title("K-means clustering on reanalysis_tdtr_k vs Total Cases")
ax[0].set_xlabel('Normalised reanalysis_tdtr_k')
ax[0].set_ylabel('Normalised Total Cases')
ax[0].legend()
ax[0].locator_params(nbins=5)

ax[1].scatter(df['reanalysis_tdtr_k'][sj],df['total_cases'][sj],color = 'indigo', label = 'San Juan',alpha =0.7)
ax[1].scatter(df['reanalysis_tdtr_k'][iq],df['total_cases'][iq], color = 'yellow', label = 'Iquitos',alpha =0.7)
ax[1].set_title(f'reanalysis_tdtr_k vs Total Cases')
ax[1].set_xlabel('reanalysis_tdtr_k')
ax[1].set_ylabel('Total Cases')
ax[1].legend()
ax[1].locator_params(nbins=4)

plt.show();

check = df['reanalysis_tdtr_k'][np.invert(np.array(labels,dtype = bool))]
check1 = df['reanalysis_tdtr_k'][np.array(labels,dtype = bool)]

n=df['city'][iq].sum()

for i in df['reanalysis_tdtr_k'][iq].index:
    for j in check.index:
        if i == j:
            n=n-1
            
m=np.array(df['city'][sj]) + 1        
m = pd.DataFrame(m).sum()

for i in df['reanalysis_tdtr_k'][sj].index:
    for j in check1.index:
        if i == j:
            m=m-1
            
score = (1199-m.values[0]-n)/1199

print(f'K-means Classification score : {100*round(score,3)}%')
print()
print("K-means does a good job of clustering the data in this example, unfortunately it does still over-classify the San Juan cluster by 41 data points.")





#%%

#decision tree on same data 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


y = df['city'].values[:]

X = pd.concat([df['reanalysis_tdtr_k'],df['total_cases']],axis = 1)
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0, test_size = 0.5)

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)

labels = tree.predict(X_test)

fig,ax = plt.subplots(1,2,figsize = (12,5))
ax[0].scatter(X_test[:,0][np.array(labels,dtype = bool)],X_test[:,1][np.array(labels,dtype = bool)], c = 'yellow',label = 'Iquitos', alpha = 0.7)
ax[0].scatter(X_test[:,0][np.invert(np.array(labels,dtype = bool))],X_test[:,1][np.invert(np.array(labels,dtype = bool))], c ='indigo', alpha = 0.7, label = 'San Juan')
ax[0].set_title("Decision tree Classification on reanalysis_tdtr_k vs Total Cases")
ax[0].set_xlabel('reanalysis_tdtr_k')
ax[0].set_ylabel('Total Cases')
ax[0].legend()
ax[0].locator_params(nbins=5)

ax[1].scatter(df['reanalysis_tdtr_k'][sj],df['total_cases'][sj],color = 'indigo', label = 'San Juan',alpha =0.7)
ax[1].scatter(df['reanalysis_tdtr_k'][iq],df['total_cases'][iq], color = 'yellow', label = 'Iquitos',alpha =0.7)
ax[1].set_title(f'reanalysis_tdtr_k vs Total Cases')
ax[1].set_xlabel('reanalysis_tdtr_k')
ax[1].set_ylabel('Total Cases')
ax[1].legend()
ax[1].locator_params(nbins=4)

plt.show();

print(tree.score(X_test,y_test))


#%%


#make group with todd
#decision tree

#put into template

#check everythings in place






