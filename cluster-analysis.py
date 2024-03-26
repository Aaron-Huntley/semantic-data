# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


def import_interoception(file_path):
    data = pd.read_csv(file_path)
    
    #A is interoception
    A = [data.Word.values,data.Interoceptive_Lancaster.values]
    
    # #remove unrated words
    A[1] = A[1].astype(float)
    float_mask = ~np.isnan(A[1])
    A[0] = A[0][float_mask]
    A[1] = A[1][float_mask]                      
                          
    return A
    
def distance_matrix(raw):
    interoceptive_ratings = raw[1]
    n = len(interoceptive_ratings)
    
    # Convert interoceptive ratings to a NumPy array
    interoceptive_ratings_np = np.array(interoceptive_ratings)
    

    # Calculate absolute differences between all pairs of interoceptive ratings
    M = np.abs(interoceptive_ratings_np[:, np.newaxis] + interoceptive_ratings_np)
    M[np.diag_indices(n)] = 0
    
    return M

    
def higherarchecal_clustering(M):
    # Creating a dendrogram
    # You can change the linkage method as per your requirement
    plt.figure(figsize=(10, 7))
    linked = linkage(M, 'single')
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Word Index')
    plt.ylabel('Distance')
    plt.show()
    
    # Performing agglomerative clustering
    #cluster = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='ward')

    # Fit and predict clusters using the precomputed distance matrix
    #labels = cluster.fit_predict(M)
    

def filter_distance(raw,d):
    #Filters the distance
    
    mask = raw[1] < d
    rawf = [[],[]]
    rawf[0]=raw[0][mask]
    rawf[1]=raw[1][mask]
    
    return rawf

def words_back(cluster,linked):
    W = []
    
    
    
    return W


intero = import_interoception("Dataset.csv")

#Truncated data
interoT = [[],[]]
interoT[0] = intero[0][:100]
interoT[1] = intero[1][:100]

#Distance filtered data
interoD = filter_distance(intero,1)

M = distance_matrix(interoD)
higherarchecal_clustering(M)



    