# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:10:25 2024

@author: ahuntle4
"""

import numpy as np
import math
import pandas as pd
import pickle


from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix

from gtda.graphs import GraphGeodesicDistance
from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, FlagserPersistence

from igraph import Graph

from IPython.display import SVG, display

def count_numeric_entries(column):
    return sum(1 for x in column if np.isfinite(x))



def load_dimentions(data):
    A = data.values       
    return True
    

# Read the CSV file
data = pd.read_csv("Dataset.csv")

# Extract numerical data excluding the first column
A_array = data.values
A_t = A_array.T

#Find indexes with less that 8k words
ind=0
ind_lis = []
for x in A_t:
    c=0
    for i in x:
        
        if i != i:
            c+=1
    if c>400:
        ind_lis.append(ind)
    ind+=1

#Minkow&Exclusivity
ind_lis.append(20)
ind_lis.append(21)
#Mohammad studies
ind_lis.append(28)
ind_lis.append(29)
ind_lis.append(30)
#AoA
ind_lis.append(32)
#Ans
ind_lis.append(55)

count = 0
for a in ind_lis:
    A_array = np.delete(A_array, a-count, axis=1)
    count+=1
    
for i in range(0,56):
    if i not in ind_lis:
        print(data.columns[i])
        
#Remove words not rated agains all groundings
count = 0
for x in A_array:
    dele = False
    for i in x:
        if i != i:
            dele = True
    if dele == True:
        A_array = np.delete(A_array, count, axis=0)
        count -=1
    count+=1
            
with open('C:/Users/ahuntle4', 'wb') as file:
    # Serialize and write the array to the file
    pickle.dump(A_array, file)




# # Convert 'nan' strings to NaN and convert to float
# A = np.array([[float('nan') if x == 'nan' else float(x) for x in row] for row in A_array])
# A_transpose = A.T

# A_new = [words.tolist()]

# for i in A:
#     A_new.append(i)
    
# A_new_array = np.array(A_new)

# Filter out columns with less than 8000 numeric entries
#A_filtered = A_array[:, np.sum(np.isnan(A), axis=0) < (len(A) - 8000)]


# data = pd.read_csv("Dataset.csv")

# valence = [data.iloc[:, 0].values,data.iloc[:, 22].values]
# V = [distance_matrix(valence)]
# arousal = [data.iloc[:, 0].values,data.iloc[:, 23].values]
# A = [distance_matrix(arousal)]

# VR = VietorisRipsPersistence(metric="precomputed")

# # Compute persistence diagrams corresponding to each entry (only one here) in X
# diagramsV = VR.fit_transform(V)
# diagramsA = VR.fit_transform(A)

# VR.plot(diagramsV)
# VR.plot(diagramsA)

# print(f"diagrams.shape: {diagramsV.shape} ({diagramsV.shape[1]} topological features)")
# print(f"diagrams.shape: {diagramsA.shape} ({diagramsA.shape[1]} topological features)")
    