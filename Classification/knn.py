#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 19:50:55 2019
@author: pabloruizruiz
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN

FEATURE_NAMES = 'resources/features.txt'
TRAIN_DATA = 'resources/train/X_train.txt'
TRAIN_LABELS = 'resources/train/y_train.txt'

feats = pd.read_table(FEATURE_NAMES, sep='\n', header=None)
har_train = pd.read_table(TRAIN_DATA, sep='\s+', header=None)
har_train_labels = pd.read_table(TRAIN_LABELS, sep='\n', header=None, names=["label"], squeeze = True)


'''
For each record it is provided:
======================================

- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
- Triaxial Angular velocity from the gyroscope. 
- A 561-feature vector with time and frequency domain variables. 
- Its activity label. 
- An identifier of the subject who carried out the experiment.

'''

har_train.columns = feats.iloc[:,0]
har_train.head()

a = har_train.isna().sum()

# Correlation

first_twenty = har_train.iloc[:, :20] # pull out first 20 feats
corr = first_twenty.corr()  # compute correlation matrix
mask = np.zeros_like(corr, dtype=np.bool)  # make mask
mask[np.triu_indices_from(mask)] = True  # mask the upper triangle

fig, ax = plt.subplots(figsize=(11, 9))  # create a figure and a subplot
cmap = sns.diverging_palette(220, 10, as_cmap=True)  # custom color map
#sns.heatmap(
#    corr,
#    mask=mask,
#    cmap=cmap,
#    center=0,
#    linewidth=0.5,
#    cbar_kws={'shrink': 0.5}
#)


# Target Variable Analysis
# ------------------------

har_train_labels.value_counts()


# Model - kNN
# ------------
 
y = har_train_labels 
X = har_train
data = pd.concat([X, y], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=24)


def euclidean_distance(x,y):
    x, y = np.array(x), np.array(y)
    return np.linalg.norm(x - y) 

p1 = (1,2,3,-4,6)
p2 = (10,2,32,-2,0)

euclidean_distance(p1,p2)

p1 = (5,5)
p2 = (0,0)
p3 = (5,6,7,8,9,10)
p4 = (1,2,3,4,5,6)
print(euclidean_distance(p1,p2)) #--> 7.0710678118654755
print(euclidean_distance(p3,p4)) #--> 9.797958971132712

        
def all_distances(test_point, data_set):
    """
    Find and return a list of distances between the "test_point"
    and all the points in "data_set", sorted from smallest to largest.
    
    Positional Arguments:
        test_point -- a Pandas Series corresponding to a row in "data_set"
        data_set -- a Pandas DataFrame
    
    Example:
        test_point = har_train.iloc[50,:]
        data_set = har_train
        
        print(all_distances(test_point, data_set)[:5])
        #--> [0.0, 2.7970187358249854, 2.922792670143521, 2.966555149052483, 3.033982453218797]
    
    """
    distances = list()
    for r in range(data_set.shape[0]):
        distances.append(euclidean_distance(test_point.values, data_set.iloc[r,:].values))
    return sorted(distances)

test_point = har_train.iloc[50,:]
data_set = har_train

print(all_distances(test_point, data_set)[:5])
#--> [0.0, 2.7970187358249854, 2.922792670143521, 2.966555149052483, 3.033982453218797]


def labels_of_smallest(numeric, labels, n):
    
    """
    Return the n labels corresponding to the n smallest values in the "numeric"
    numpy array.
    
    Positional Arguments:
        numeric -- a numpy array of numbers
        labels -- a numpy array of labels (string or numeric)
            corresponding to the values in "numeric"
        n -- a positive integer
        
    Example:
        numeric = np.array([7,6,5,4,3,2,1])
        labels = np.array(["a","a","b","b","b","a","a"])
        n = 6
        
        print(labels_of_smallest(numeric, labels, n))
        #--> np.array(['a', 'a', 'b', 'b', 'b', 'a'])
    """
    frame = pd.DataFrame(np.array((numeric, labels)).T, columns=['num','lab'])
    frame.sort_values(by='num', inplace=True)
    return np.array(frame['lab'].values[:n])

numeric = np.array([7,6,5,4,3,2,1])
labels = np.array(["a","a","b","b","b","a","a"])
n = 6

print(labels_of_smallest(numeric, labels, n))
#--> np.array(['a', 'a', 'b', 'b', 'b', 'a'])

# frame = pd.DataFrame(np.array((numeric, labels)).T, columns=['num','lab'])


def label_voting(labels):
    """
    Given a numpy array of labels. Return the label that appears most frequently
    If there is a tie for most frequent, return the label that appears first.
    
    Positional Argument:
        labels -- a numpy array of labels
    
    Example:
        lab1 = np.array([1,2,2,3,3])
        lab2 = np.array(["a","a","b","b","b"])
        
        print(label_voting(lab1)) #--> 2
        print(label_voting(lab2)) #--> "b"
        
    """
    from collections import Counter

    c = Counter(labels)
    all_ = c.most_common()
    most = c.most_common()[0][1]
    coms = [a for a in all_ if a[1] == most]
    
    if len(coms) > 1:    
        candidates = [a[0] for a in coms]
        for r in labels:
            if r in candidates:
                return r
    else:
        return coms[0][0]

lab1 = np.array([1,2,2,3,3])
lab2 = np.array(["a","a","b","b","b"])

print(label_voting(lab1)) #--> 2
print(label_voting(lab2)) #--> "b"


def custom_KNN( point, X_train, y_train, n):
    """
    Predict the label for a single point, given training data and a specified
    "n" number of neighbors.
    
    Positional Arguments:
        point -- a pandas Series corresponding to an observation of a point with
             unknown label.
        x_train -- a pandas DataFrame corresponding to the measurements
            of points in a dataset. Assume all values are numeric, and
            observations are in the rows; features in the columns
        y_train -- a pandas Series corresponding to the labels for the observations
            in x_train
    
    Example:
        point = pd.Series([1,2])
        X_train = pd.DataFrame([[1,2],[3,4],[5,6]])
        y_train = pd.Series(["a","a","b"])
        n = 2
        print(custom_KNN(point, X_train, y_train, n)) #--> 'a'
    """
    # 1 - Calculate distances between test point and every point
    distances = all_distances(point, X_train)
    # 2 - Finds the labels from the "n" nearest neighbors
    labels = labels_of_smallest(distances, y_train, n)
    # 3 - Returns a prediction according to the voting rules 
    pred = label_voting(labels)
    return pred


point = pd.Series([1,2])
X_train = pd.DataFrame([[1,2],[3,4],[5,6]])
y_train = pd.Series(["a","a","b"])
n = 2
print(custom_KNN(point, X_train, y_train, n)) #--> 'a'

## TODO: check if all_distances have to return the SORTED version of the array


