# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd


def import_grounding(data,grounding):
    """Input data matrix from csv file and a string for which grounding.
    Output raw data as a list of two arrays"""
    
    '''NEED TO NORMALISE THIS'''
    
    if grounding == "Interoceptive":
        #0-5
        A = [data.Word.values,data.Interoceptive_Lancaster.values]
    elif grounding == "Socialness":
        #1-7
        A = [data.Word.values,data.Socialness.values]
    elif grounding == "Valence_Warriner":
        #0-9
        A = [data.Word.values,data.Valence_Warriner.values]
    elif grounding == "Arousal_Warriner":
        #0-9
        A = [data.Word.values,data.Arousal_Warriner.values]
        
    #remove unrated words
    A[1] = A[1].astype(float)
    float_mask = ~np.isnan(A[1])
    A[0] = A[0][float_mask]
    A[1] = A[1][float_mask]                      
                          
    return A
    
def distance_matrix(raw):
    """Input raw data as a list of two arrays.
    Output a distance matrix based on the second array"""
    
    interoceptive_ratings = raw[1]
    n = len(interoceptive_ratings)
    
    # Convert interoceptive ratings to a NumPy array
    interoceptive_ratings_np = np.array(interoceptive_ratings)
    
    # Calculate absolute differences between all pairs of interoceptive ratings
    M = np.abs(interoceptive_ratings_np[:, np.newaxis] + interoceptive_ratings_np)
    M[np.diag_indices(n)] = 0
    
    return M
    
def higherarchecal_clustering(M,D):
    """Input distance matrix and a bool.
    Return the (single) linked data structure based on M.
    Output a dendogram based on linked if D is True"""
    
    plt.figure(figsize=(10, 7))
    linked = linkage(M, 'single')
    
    if D == True:
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Dendrogram')
        plt.xlabel('Word Index')
        plt.ylabel('Distance')
        plt.show()
    
    return linked

def filter_distance(raw,d):
    """Input raw data as a list of two arrays and a float, d. 
    Return list of two sets only including entries at positions whose distance
    is less than d"""
    
    mask = raw[1] < d
    rawf = [[],[]]
    rawf[0]=raw[0][mask]
    rawf[1]=raw[1][mask]
    
    return rawf

def jaccard_similarity(set1,set2):
    """Input two sets.
    Return their similarity percentage as a float.
    The similarity percentage is the intersection over the union times 100"""
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union if union != 0 else 0  # Handle division by zero
    return similarity * 100

def percent_compare(G1,G1N,G2,G2N,D):
    """Input two raw data as a list of two arrays (G1,G2) and their names as 
    a string (G1N,G2N) and a bool (D).
    Output a list of size 1000 where entry n is the jaccard similarity of
    the words of distance less than 8n/1000"""

    combined_G1 = list(zip(G1[0],G1[1]))
    ordered_pairs_G1 = sorted(combined_G1, key=lambda x: x[1])
    ordered_strings_G1, ordered_floats_G1 = zip(*ordered_pairs_G1)
    
    combined_G2 = list(zip(G2[0],G2[1]))
    ordered_pairs_G2 = sorted(combined_G2, key=lambda x: x[1])
    ordered_strings_G2, ordered_floats_G2 = zip(*ordered_pairs_G2)
    
    #Assuming the normalised data is 0-8  
    x = np.linspace(0, 8, 1000)
    y=[]
    for i in x:
        G1Str = [string for string, value in ordered_pairs_G1 if value < i]
        G2Str = [string for string, value in ordered_pairs_G2 if value < i]
        y.append(jaccard_similarity(set(G1Str),set(G2Str)))

    if D == True:
        plt.plot(x,y)
        plt.title('Comparison of '+str(G1N)+" and "+str(G2N))
        plt.xlabel('Words of distance < x')
        plt.ylabel('Percent Similarity')
        plt.show()
            
    return y

def combine_raw(raw1,raw2,H):
    """Input two groundings and combine them using the hyperparemeter H.
    Return the combined grounding as a list with two arrays.
    Warning: doesn't check if the raw data has the same word ratings"""
    
    combined = [[],[]]
    if H == "min":
        combined[0] = raw1[0]
        combined[1] = [min(a, b) for a, b in zip(raw1[1], raw2[1])]
        return combined
    elif H == "max":
        combined[0] = raw1[0]
        combined[1] = [max(a, b) for a, b in zip(raw1[1], raw2[1])]
        return combined
    elif H == "mean":
        combined[0] = raw1[0]
        combined[1] = [(a+b)/2 for a, b in zip(raw1[1], raw2[1])]
        return combined



''''TEST RUNNING'''

data = pd.read_csv("Dataset.csv")

'''Import Data'''

valence = import_grounding(data,"Valence_Warriner")
arousal = import_grounding(data,"Arousal_Warriner")
social = import_grounding(data,"Socialness")
intero = import_grounding(data,"Interoceptive")

infinite =[[],[]]
infinite[0] = social[0]
infinite[1] = np.zeros(len(social[0]))

valenceT = [[],[]]
valenceT[0] = valence[0][:10]
valenceT[1] = valence[1][:10]

arousalT = [[],[]]
arousalT[0] = arousal[0][:10]
arousalT[1] = arousal[1][:10]

'''Test dendogram'''

# intero = import_grounding(data,"Interoceptive")

# #Truncated data
# interoT = [[],[]]
# interoT[0] = intero[0][:100]
# interoT[1] = intero[1][:100]

# #Distance filtered data
# interoD = filter_distance(intero,1)

# M = distance_matrix(interoD)
# interolinked  = higherarchecal_clustering(M,False)

'''Test percentage graphs'''

# Val_Aro = percent_compare(valence,"Valence",arousal,"Arousal",True)
# Val_Soc = percent_compare(valence,"Valence",social,"Social",True)

# Aro_Soc = percent_compare(arousal,"Arousal",social,"Social",True)
# Aro_Int = percent_compare(arousal,"Arousal",intero,"Intero",True)

# Val_Int = percent_compare(valence,"Valence",intero,"Intero",True)
# Soc_Int = percent_compare(social,"Social",intero,"Intero",True)

# Aro_Aro = percent_compare(arousal,"Arousal",arousal,"Arousal",True)

'''Test combined'''

emotion = combine_raw(valence, arousal, "min")

# Emo_Int = percent_compare(emotion,"Emotion",intero,"Intero",True)

Inf_Int =  percent_compare(infinite,"Inf",intero,"Intero",True)



    