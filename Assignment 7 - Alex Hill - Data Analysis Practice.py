# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:49:18 2019

@author: alex
"""

#Data Analysis Assignment 7


#%%

#Question: Are a one-sample t test and a paired t test equivalent? Why or why not?
#
#    Answer: They are fundamentally different as a one-sample t test is testing
#            whether the sample mean is equal to some pre-existing suspected
#            value, where a paired t test is testing whether two dependent 
#            samples have the same expected value(mean) or not.
#
#Question: What is the difference between two-tailed and one-tailed results?
#
#    Answer: The core difference between a two-tailed test and a one-tailed
#            test is that they correspong to two different null hypotheses.
#            The two-tailed test corresponds to a Null Hypothesis : H_0 =/= Mu,
#            whereas a one-tailed test corresonds to H_0 >= Mu or H_0 <= Mu, 
#            where Mu is the suspected value for the means of the sample. The 
#            reason we talk of 'tails' is because when we look at the t-
#            distrubution it has two tails on either side, and when we want to
#            test whether H_0 >= Mu or H_0 <= Mu, we only need to look at one
#            tail of the distrubution, where  H_0 =/= Mu you look at both tails.

#%%

# t statistic calculations

# Task: Use np.mean and np.std to compute the t statistic value for the 
#       "One-sample t test" example from the Lecture07 notes (see Equation 1).

import numpy as np
from scipy import stats

data       = np.array([23, 15, -5, 7, 1, -10, 12, -8, 20, 8, -2, -5])
mu      = 0
results = stats.ttest_1samp(data, mu)

tstat = np.mean(data)/(np.std(data,ddof=1)/np.sqrt(12))
tstat1 = results[0]


print("----------------------------------")
print()
print("Task 1 : One-Sample t test")
print()
print("Numerical result :",round(tstat,4))
print()
print("Analytical result :",round(tstat1,4))
print()
print("The Numerical and Analytical values are the same :" , tstat1 == tstat)
print()
print("----------------------------------")
#%%

# Task: Use np.mean and np.std to compute the t statistic value for the
#       "Paired t test" example from the Lecture07 notes (see Equation 2).

from scipy.stats import t

y_pre  = np.array( [3, 0, 6, 7, 4, 3, 2, 1, 4] )
y_post = np.array( [5, 1, 5, 7, 10, 9, 7, 11, 8] )

tana,pana    = stats.ttest_rel(y_pre, y_post)
std = np.std(y_pre-y_post,ddof=1)
stder = std/np.sqrt(len(y_post))

summ =0

for i in range(0,len(y_pre)):
    summ = summ + (y_pre[i]-y_post[i])
    dmean = summ/len(y_pre)
tnum = dmean/stder

print()
print("Task 2 : Paired t test")
print()
print("Numerical result :",round(tnum,4))
print()
print("Analytical result :",round(tana,4))
print()
print("The Numerical and Analytical values are the same :" , tnum == tana)
print()
print("----------------------------------")

#%%

# Task: Use np.mean and np.std to compute the t statistic value for the
#      "Two-sample t test" example from the Lecture07 notes (see Equation 3).


beginning = np.array( [3067, 2730, 2840, 2913, 2789] )
end       = np.array( [3200, 2777, 2623, 3044, 2834] )

tana,pana    = stats.ttest_ind(beginning, end)

begmean = np.mean(beginning)
endmean = np.mean(end)


n=5

sp = np.sqrt(((n-1)*np.std(beginning,ddof=1)**2+(n-1)*np.std(end,ddof=1)**2)/8)

tnum = (begmean-endmean)/(sp*np.sqrt(2/5))
print()
print("Task 3 : Two Sample t test")
print()
print("Numerical result :",round(tnum,4))
print()
print("Analytical result :",round(tana,4))
print()
print("The Numerical and Analytical values are the same :" , tnum == tana)
print()
print("----------------------------------")

#%%

# Dataset analyses

# Dataset 1

old = np.array([44,49,56,51,38,44,61,51,49,60,39,51,43,37,45])
new = np.array([51,42,37,45,47,65,49,69,38,44,49,56,51,50,38])


type1 = "Two Sample (Two tail)"

t1,p1    = stats.ttest_ind(old, new)

if p1 < 0.05:
    nullregect1 = "Yes"
else:
    nullregect1 = "No"

##############################################

# Dataset 2

data =np.array([ 18, 22, 
                 21 ,25, 
                 16 ,17, 
                 22 ,24,
                 19 ,16, 
                 24 ,29, 
                 17 ,20, 
                 21 ,23,
                 23 ,19,
                 18 ,20,
                 14 ,15,
                 16 ,15,
                 16 ,18,
                 19 ,26,
                 18 ,18,
                 20 ,24,
                 12 ,18,
                 22 ,25,
                 15 ,19, 
                 17 ,16])
    
predata = []
postdata = []

for i in range(0,len(data)):
    if i % 2 ==0:
        predata = np.append(predata,data[i])
    else:
        postdata = np.append(postdata,data[i])

type2 = "Paired (One tail)"

t2,p2   = stats.ttest_rel(y_pre, y_post)
p2 = p2/2  #One tail

if p2 < 0.05:
    nullregect2 = "Yes"
else:
    nullregect2 = "No"

##############################################

# Dataset 3
    
data3 = np.array([
                128 ,127,
                118	,115,
                144	,142,
                133	,140,
                132	,131,
                111	,132,
                149	,122,
                139	,119,
                136	,129,
                126	,128])

mu      = 120
results = stats.ttest_1samp(data3, mu)
p3 = results[1]
t3 = results[0]
type3 = "One Sample (Two tail)"

if p3 < 0.05:
    nullregect3 = "Yes"
else:
    nullregect3 = "No"

##############################################
    
# Dataset 4
    
before = np.array([135,142,137,122,147,151,131,117,154,143,133])
after = np.array([127,145,131,125,132,147,119,125,132,139,122])

type4 = "Paired (One tail)"

t4,p4   = stats.ttest_rel(before, after)
p4 = p4/2   #One tail

if p4 < 0.05:
    nullregect4 = "Yes"
else:
    nullregect4 = "No"
    
##############################################
    
# Dataset 5
    
type5 = "One Sample (Two tail)"
    
data5 = np.array([240,243,250,254,264,279,284,285,290,298,
                  302,310,312,315,322,337,348,384,386,520])
    
mu      = 200
results = stats.ttest_1samp(data5, mu)
p5 = results[1]
t5 = results[0]

if p5 < 0.05:
    nullregect5 = "Yes"
else:
    nullregect5 = "No"
    
##############################################
    

#Lets make the table

print("{0:^13}|{1:^23}|{2:^8}|{3:^8}|{4:^13}|".format("Dataset","Test Type","t","p"," H0 Rejected?"))
print("----------------------------------------------------------------------")
print("{0:^13}|{1:^23}|{2:^8.3f}|{3:^8.3f}|{4:^13}|".format("1",type1,t1,p1,nullregect1))
print("{0:^13}|{1:^23}|{2:^8.3f}|{3:^8.3f}|{4:^13}|".format("2",type2,t2,p2,nullregect2))
print("{0:^13}|{1:^23}|{2:^8.3f}|{3:^8.3f}|{4:^13}|".format("3",type3,t3,p3,nullregect3))
print("{0:^13}|{1:^23}|{2:^8.3f}|{3:^8.3f}|{4:^13}|".format("4",type4,t4,p4,nullregect4))
print("{0:^13}|{1:^23}|{2:^8.3f}|{3:^8.3f}|{4:^13}|".format("5",type5,t5,p5,nullregect5))

    
    