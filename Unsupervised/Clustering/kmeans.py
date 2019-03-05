#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:14:15 2019
@author: pabloruizruiz
"""

import numpy as np

'''

## Question 1 - Building Cluster Assignment Algorithms

For the point ii; set cluster indicator cici to be the kk according to the equation 
ci=argmink ||xi−μk||2ci=argmink ||xi−μk||2, where kk is one of the clusters.

e.g. The point ii is in the cluster with the center nearest to the point.

Code a function called assign_clusters_k_means Accept two arguments:

    points: a 2-d numpy array of the locations of each point
    clusters: a 2-d numpy array of the locations of the centroid of each cluster.
    Determine which cluster centroid is closest to each point.

RETURN a 2-d numpy array where each row indicates which cluster a point is closets to, and thus also assigned to:

e.g. [0,1,0,...,0] indicates the point is assigned to the second cluster, and
[0,0,...,1] indicates the point is assigned to the last cluster
'''


def assign_clusters_k_means(points, clusters):
    """
    Determine the nearest cluster to each point, returning an array indicating the closest cluster
    
    Positional Arguments:
        points: a 2-d numpy array where each row is a different point, and each
            column indicates the location of that point in that dimension
        clusters: a 2-d numpy array where each row is a different centroid cluster;
            each column indicates the location of that centroid in that dimension
    """
    # NB: "cluster_weights" is used as a common term between functions
    # the name makes more sense in soft-clustering contexts
    def distance(p,c):
        return ((p[0] - c[0])**2 + (p[1] - c[1])**2)**0.5
            
    cluster_weights = np.zeros((len(points), len(clusters)))
    for i,point in enumerate(points):
        d = [distance(point, cluster) for cluster in clusters]
        cluster_weights[i][np.argmin(d)] = 1
    
    return cluster_weights



points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
clusters = np.array([[0,1],[5,4]])
cluster_weights = assign_clusters_k_means(points, clusters)

print(cluster_weights) #--> np.array([[1, 0],
#                                      [1, 0],
#                                      [0, 1],
#                                      [0, 1],
#                                      [0, 1]])




def assign_clusters_soft_k_means(points, clusters, beta):
    """
    Return an array indicating the porportion of the point
        belonging to each cluster
    
    Positional Arguments:
        points: a 2-d numpy array where each row is a different point, and each
            column indicates the location of that point in that dimension
        clusters: a 2-d numpy array where each row is a different centroid cluster;
            each column indicates the location of that centroid in that dimension
        beta: a number indicating what distance can be considered "close"
    """
    
    def distance(p,c):
        return ((p[0] - c[0])**2 + (p[1] - c[1])**2)**0.5
      
    denominators = [np.sum([ np.exp((-1/beta)*distance(p,c)) for c in clusters ]) for p in points]
        
    for i, point in enumerate(points):
        cluster_weights[i] = [np.exp((-1/beta)*distance(point,c)) / denominators[i] for c in clusters]

    return cluster_weights


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
clusters = np.array([[0,1],[5,4]])
beta = 1
cluster_weights = assign_clusters_soft_k_means(points, clusters, beta)
print(cluster_weights) #--> np.array([[0.99707331, 0.00292669],
#                                      [0.79729666, 0.20270334],
#                                      [0.00292669, 0.99707331],
#                                      [0.04731194, 0.95268806],
#                                      [0.1315826 , 0.8684174 ]])