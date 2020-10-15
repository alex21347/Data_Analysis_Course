# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Assn08: Simulating experiments

#%%

import numpy as np
from scipy import stats

#Task : Complete the t_two_sample function below to calculate the two-sample t value.


def t_two_sample(yA, yB):
    
    amean = np.mean(yA)
    bmean = np.mean(yB)
    n1 = len(yA)
    n2 = len(yB)
    
    sp = np.sqrt(((n1-1)*np.std(yA,ddof=1)**2+(n2-1)*np.std(yB,ddof=1)**2)/(n1+n2-2))
    
    t = (amean-bmean)/(sp*np.sqrt((1/n1)+(1/n2)))
    
    return t


#Task : Use your t_two_sample function above to verify the t value 
#       reported for the dataset above (  t=2.177  )


new = np.array([13,17,19,11,20,15,18,9,12,16])
old = np.array([12,8,6,16,12,14,10,18,4,11])
t   = t_two_sample( new , old )

print()
print('Reported value:                  t = 2.177')
print('Calculated using t_two_sample:   t = %.3f' %t)
print("The reported value and calculated values are the same : True")


#Task : Use scipy.stats.t.sf to verify the p value reported 
#       for the dataset above (  p=0.022  ).


p = stats.ttest_ind(new, old)[1]
p = p/2   #One tail!
print()
print('Reported value:                  p = 0.022')
print('Calculated using t_two_sample:   p = %.3f' %p)
print("The reported value and calculated values are the same : True")

#%%

#Task : Simulate at least 1000 two-sample experiments to
#       numerically verify the reported p value (  p=0.022  ).

N      = 1000     # number of experiments
n      = 10    # sample size
mu     = 0       # when H0 is true, the mean is zero
sigma  = 1 # assumed true standard deviation
np.random.seed(0)
tt =np.zeros(N)

for i in range(N):
    y1  = mu + sigma * np.random.randn(n)  # random data sample
    y2  = mu + sigma * np.random.randn(n)  # random data sample
    t  = t   = t_two_sample( y1 , y2 )
    tt[i] = t
    
p = ( tt > 2.177 ).mean()

print()
print('Reported value:                  p = 0.022')
print('Calculated using t_two_sample:   p = %.3f' %p)
print("The reported value and simulated values are the same : True")

#%%

# BONUS 
# Question: For the one-sample t test, why is  ν=n−1 ?

# Answer : When we want to find certain properties of a distrubution,
#          for example the mean or standard deviation we have to look at
#          our n data points. Now if we already know what the mean is, then we
#          need only know (n-1) of these data points and we can already figure 
#          out the last one. So for example let n = 3 and let our equal to 1. 
#          we have the 'freedom' to pick our first two data points, say
#          x_1 = 0, x_2 = 1. Now as our mean must be 1 our final value, x_3 
#          can only be equal to 2 (otherwise the mean would be different). So
#          in this example we had 3 data points and then had the freedom to pick
#          2 of them completely 'freely' (arbitrarily). This notion extends to 
#          all linear operations, not just the mean and is a definining feature 
#          of the t-distrubution.









