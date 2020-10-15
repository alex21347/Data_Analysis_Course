# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Assignment 10 - Alex Hill - Data Analysis Practice II



#%%

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append('C:/Users/alex/Documents/KyotoU/Data Analysis Practice II')
import lecture12util as lu

x = np.loadtxt('C:/Users/alex/Documents/KyotoU/Data Analysis Practice II/Assn10-dataset-with-labels.csv',delimiter =',',dtype = float)


#%%

# Part 1:   Unsupervised learning

#This plots labeled data
# =============================================================================
# plt.figure()
# ax = plt.axes()
# for i in range(0,len(x[:,0])):  
#     ax.plot(x[i,0], x[i,1], 'o',c = (x[i,2]/2,0.5+x[i,2]/2,0.5+x[i,2]/2))
# ax.set_xlabel('x0', size=16)
# ax.set_ylabel('x1', size=16)
# ax.axis('equal')
# plt.show()
# =============================================================================

plt.figure()
ax = plt.axes()
ax.plot(x[:,0], x[:,1], 'o')
ax.set_xlabel('x0', size=16)
ax.set_ylabel('x1', size=16)
ax.axis('equal')
plt.show()
#%%


# =============================================================================
# 
# Run unsupervised learnign on this dataset.
# 
# Use K-Means Clustering with two clusters.
# 
# Use lu.plot_labeled_points to plot the results.
# 
# =============================================================================


kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(x);
labels = kmeans.labels_
lu.plot_labeled_points(x[:,:2], labels, colors=['r','b'])


#Question 1.1
#Are these clusters well separated?

#Answer : No

dw,db   = lu.average_distances(x, labels, ratios=True)
print(dw)

#As the average within-cluster distances are almost the same size as the distance
#between the centroids, this implies the two groups are not well seperated.


#%%


#Part 2:   Supervised learning

labels = x[:,2]

#Question 2.1
#How many unique labels does this dataset have?

#Answer = 2

#Task 2.2

lu.plot_labeled_points(x[:,:2],labels, colors=['r','b'])


#Question 2.2
#Is there more overlap between the clusters than the K-Means results?    

#Answer  = Yes

#Task 2.3

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x[:,:2], labels);
#%%
plt.figure( figsize=(8,6) )
lu.plot_decision_surface(knn,colors=['b','r'], n=50, alpha=0.3, marker_size=200, marker_alpha=0.7)

#Question 2.3
#Does the decision boundary look linear, like in the K-Means analysis?

#Answer = No

#%%

#Question 2.4
#What is the classification rate for this fitted KNN classifier?
labelsp = knn.predict(x[:,:2])
cr = lu.classification_rate(labels, labelsp)
print(cr)


#This classification rate is very high, primarily because it uses the training set
#as also a test set.





