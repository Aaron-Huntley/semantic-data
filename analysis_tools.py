# -*- coding: utf-8 -*-
"""
Tools for data analysis:
    
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import os


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

def percent_compare_linear(G1,G1N,G2,G2N,D,pathname):
    """Input two linear normalised groundings as a list of two arrays (G1,G2)
    and their names as 
    a string (G1N,G2N) and a bool (D).
    renormalise both groundings to only compare words with ratings on both 
    groundings.
    Output a list of size 1000 where entry n is the jaccard similarity of
    the words of distance less than 8n/1000.
    Saves plots at pathname creates a folder named after the first grounding
    if there is not one already there."""


    combined_G1 = list(zip(G1[0],G1[1]))
    ordered_pairs_G1 = sorted(combined_G1, key=lambda x: x[1])
    G1[0], G1[1] = zip(*ordered_pairs_G1)
    
    combined_G2 = list(zip(G2[0],G2[1]))
    ordered_pairs_G2 = sorted(combined_G2, key=lambda x: x[1])
    G2[0], G2[1] = zip(*ordered_pairs_G2)
    
    common_words = set(G1[0]).intersection(set(G2[0]))
    
    
    mask1 = [item in common_words for item in G1[0]]
    mask2 = [item in common_words for item in G2[0]]

    # Apply masks to the lists
    ordered_pairs_G1 = [item for item, m in zip(ordered_pairs_G1, mask1) if m]
    ordered_pairs_G1 = [item for item, m in zip(ordered_pairs_G2, mask2) if m]
    
    #Assuming the normalised data is 0-8  
    x = np.linspace(0, 8, 1000)
    y=[]
    for i in x:
        G1Str = [string for string, value in ordered_pairs_G1 if value < i]
        G2Str = [string for string, value in ordered_pairs_G2 if value < i]
        y.append(jaccard_similarity(set(G1Str),set(G2Str)))

    if D == True:
        plt.plot(x,y)
        plt.title('Comparison of '+str(G1N)+" and "+str(G2N)+"(linear)")
        plt.xlabel('Words of distance < x')
        plt.ylabel('Percent Similarity')
        if not os.path.exists(pathname+'/'+str(G1N)):
            os.makedirs(pathname+'/'+str(G1N))
        plot_path = pathname+'/'+str(G1N)+'/'+str(G1N)+" and "+str(G2N)+'(ord).png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()
            
    return y


def percent_compare_ordered(G1,G1N,G2,G2N,D,pathname):
    """Input two ordered normalised grouindings and their names. 
    D a bool True if you want plots. 
    pathname is a string of where you want to save the plots
    saves the plots at the path name and creates a folder named after the first
    ordered list if there is not one there."""
    
    common_words = set(G1).intersection(set(G2))
    mask1 = [item in common_words for item in G1]
    mask2 = [item in common_words for item in G2]
    
    G2 = [item for item, m in zip(G2, mask2) if m]
    G1 = [item for item, m in zip(G1, mask1) if m]
    
    x = np.arange(max(len(G1),len(G2)))
    y=[]
    for i in x:
        y.append(jaccard_similarity(set(G1[:i]),set(G2[:i])))

    if D == True:
        plt.plot(x,y)
        plt.title('Comparison of '+str(G1N)+" and "+str(G2N)+"(ordered)")
        plt.xlabel('First n words in ordered list')
        plt.ylabel('Percent Similarity')
        if not os.path.exists(pathname+'/'+str(G1N)):
            os.makedirs(pathname+'/'+str(G1N))
        plot_path = pathname+'/'+str(G1N)+'/'+str(G1N)+" and "+str(G2N)+'(ord).png'
        plt.savefig(plot_path, bbox_inches='tight')
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
    
def normalise_sort(G1):
    """Input a grounding. removes unrated words. 
    Sort based on ratings and then forget the ratings.
        Return a single sorted list of words"""
        
    G1[1] = G1[1].astype(float)
    float_mask = ~np.isnan(G1[1])
    G1[0] = G1[0][float_mask]
    G1[1] = G1[1][float_mask]
        
    combined_G1 = list(zip(G1[0],G1[1]))
    ordered_pairs_G1 = sorted(combined_G1, key=lambda x: x[1])
    ordered_strings_G1, ordered_floats_G1 = zip(*ordered_pairs_G1)    
 
    return ordered_strings_G1

def linear_transform(value,A,B,a,b):
    """Linearly transforms an interval [A,B] to [a,b]"""
    
    return (value-A)*(b-a)/B-A +a

def normalise_linear(G1,A,B,a,b):
    """input a grounding and its grading scale A to B. 
    remove all NaN values from both arrays in G1.
    output normalised data
    on the scale a to b"""
    
    G1[1] = G1[1].astype(float)
    float_mask = ~np.isnan(G1[1])
    G1[0] = G1[0][float_mask]
    G1[1] = G1[1][float_mask] 
    
    array_to_transform = G1[1]
    transformed_array = [linear_transform(value,A,B,a,b) for value in array_to_transform]
    G1[1] = transformed_array
 
    return G1
    




    