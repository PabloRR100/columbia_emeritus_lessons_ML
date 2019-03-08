#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:14:15 2019
@author: pabloruizruiz
"""

import numpy as np

## PART 1 - CLUSTER ASSIGNMENT
# 1.1 Hard assignment
# 1.2 Soft assignment
# 1.3 GMM assignment


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
        
    cluster_weights = np.zeros((len(points), len(clusters)))
    for i, point in enumerate(points):
        cluster_weights[i] = [np.exp((-1/beta)*distance(point,c)) / denominators[i] for c in clusters]

    return np.array(cluster_weights)


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
clusters = np.array([[0,1],[5,4]])
beta = 1
cluster_weights = assign_clusters_soft_k_means(points, clusters, beta)
print(cluster_weights) #--> np.array([[0.99707331, 0.00292669],
#                                      [0.79729666, 0.20270334],
#                                      [0.00292669, 0.99707331],
#                                      [0.04731194, 0.95268806],
#                                      [0.1315826 , 0.8684174 ]])


from scipy.stats import multivariate_normal
def assign_clusters_GMM(points, clusters):
    """    
    Return an array indicating the porportion of the point
        belonging to each cluster
    
    Positional Arguments:
        points: a 2-d numpy array where each row is a different point, and each
            column indicates the location of that point in that dimension
        clusters: a list of tuples. Each tuple describes a cluster.
            The first element of the tuple is a 1-d numpy array indicating the
                location of that centroid in each dimension
            The second element of the tuple is a number, indicating the weight (pi)
                of that cluster
            The thrid element is a 2-d numpy array corresponding to that cluster's
                covariance matrix.
    """
    def probabiliy(mu,Sigma,x): 
        return multivariate_normal(mu,Sigma).pdf(x)
    
    cluster_weights = np.zeros((len(points), len(clusters)))
    denominators = [np.sum([ c[1]*(probabiliy(c[0],c[2],p)) for c in clusters ]) for p in points]
            
    for i, point in enumerate(points):    
        cluster_weights[i] = [c[1]*(probabiliy(c[0],c[2],point)) / denominators[i] for c in clusters]
    
    return np.array(cluster_weights)

points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
clusters = [(np.array([0,1]), 1, np.array([[1,0],[0,1]])),
            (np.array([5,4]), 1, np.array([[1,0],[0,1]]))]

cluster_weights = assign_clusters_GMM(points, clusters)

print(cluster_weights) #--> np.array([[9.99999959e-01 4.13993755e-08]
#                                      [9.82013790e-01 1.79862100e-02]
#                                      [4.13993755e-08 9.99999959e-01]
#                                      [2.26032430e-06 9.99997740e-01]
#                                      [2.47262316e-03 9.97527377e-01]])



## PART 2 - CLUSTER UPDATE
# 1.1 Hard update
# 1.2 Soft update
# 1.3 GMM update



def update_clusters_k_means(points, cluster_weights):
    """
    Update the cluster centroids via the k-means algorithm
    
    Positional Arguments -- 
        points: a 2-d numpy array where each row is a different point, and each
            column indicates the location of that point in that dimension
        cluster_weights: a 2-d numy array where each row corresponds to each row in "points"
            and the columns indicate which cluster the point "belongs" to - a "1" in the kth
            column indicates belonging to the kth cluster    
    """
    point_dim, num_clusters = points.shape[1], cluster_weights.shape[1]
    points_to_each_cluster = [np.sum(cluster_weights[:,c]) for c in range(num_clusters)]
    
    new_centroids = np.zeros((num_clusters, point_dim))
    for c in range(num_clusters):
        w = cluster_weights[:,c]
        for i,p in enumerate(points):
            new_centroids[0][c] += np.sum(p[0]) * w[i]
            new_centroids[1][c] += np.sum(p[1]) * w[i]
    
    new_centroids /= points_to_each_cluster
    return new_centroids


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
cluster_weights = np.array([[1, 0],[1, 0],[0, 1],[0, 1],[0, 1]])

new_cents = update_clusters_k_means(points, cluster_weights)

print(new_cents) #--> np.array([[1. , 1.5],
#                                [4. , 4. ]])



def update_clusters_soft_k_means(points, cluster_weights):
    """
    Update cluster centroids according to the soft k-means algorithm
    
    Positional Arguments --
        points: a 2-d numpy array where each row is a different point, and each
            column indicates the location of that point in that dimension
        cluster_weights: a 2-d numpy array where each row corresponds to each row in 
            "points". the values in that row corresponding to the amount that point is associated
            with each cluster.
    """
    point_dim, num_clusters = points.shape[1], cluster_weights.shape[1]
    denominators = [np.sum(cluster_weights[:,c]) for c in range(num_clusters)]
    
    new_centroids = np.zeros((num_clusters, point_dim))
    for c in range(num_clusters):
        for i,p in enumerate(points): 
            new_centroids[0][c] += np.sum(p[0]) * cluster_weights[i][c]
            new_centroids[1][c] += np.sum(p[1]) * cluster_weights[i][c]
    
    new_centroids /= denominators
    return new_centroids


points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])

cluster_weights= np.array([[0.99707331, 0.00292669],
                           [0.79729666, 0.20270334],
                           [0.00292669, 0.99707331],
                           [0.04731194, 0.95268806],
                           [0.1315826 , 0.8684174 ]])
centroids = update_clusters_soft_k_means(points, cluster_weights)

print(centroids) #--> np. array([[1.15246591, 1.59418291],
#                                 [3.87673553, 3.91876291]])

def update_clusters_GMM(points, cluster_weights):
    
    """
    Update cluster centroids (mu, pi, and Sigma) according to GMM formulas
    
    Positional Arguments --
        points: a 2-d numpy array where each row is a different point, and each
            column indicates the location of that point in that dimension
        cluster_weights: a 2-d numpy array where each row corresponds to each row in 
            "points". the values in that row correspond to the amount that point is associated
            with each cluster.        
    """
    N = points.shape[0]
    point_dim, num_clusters = points.shape[1], cluster_weights.shape[1]
    new_clusts = []
    
    for c in range(num_clusters):
        
        nk = np.sum(cluster_weights[:,c])
        pi = nk / N
        
        mu_x, mu_y = 0, 0
        for p in points:
            mu_x += p[0] * cluster_weights[:,c]
            mu_y += p[1] * cluster_weights[:,c]    
        mu = 1/nk * np.array([np.sum(mu_x), np.sum(mu_y)])
        
        sig = 0
        for p in points:            
            sig = 1/nk * cluster_weights[:,c] * np.matmul((p - mu), (p - mu).T)
        
        new_clusts.append(mu, pi, sig)
    return new_clusts

points = np.array([[0,1], [2,2], [5,4], [3,6], [4,2]])
cluster_weights = np.array([[9.99999959e-01, 4.13993755e-08],
                            [9.82013790e-01, 1.79862100e-02],
                            [4.13993755e-08, 9.99999959e-01],
                            [2.26032430e-06, 9.99997740e-01],
                            [2.47262316e-03, 9.97527377e-01]])

new_clusters = update_clusters_GMM(points, cluster_weights)

print(new_clusters)
##-->[(array([0.99467691, 1.49609648]), #----> mu, centroid 1
#    0.3968977347767351, #-------------------> pi, centroid 1
#    array([[1.00994319, 0.50123508],
#           [0.50123508, 0.25000767]])), #---> Sigma, centroid 1
#    
#    (array([3.98807155, 3.98970927]), #----> mu, centroid 2
#    0.603102265479875, #-------------------> pi, centroid 2
#    array([[ 0.68695286, -0.63950027], #---> Sigma centroid 2
#           [-0.63950027,  2.67341935]]))]
