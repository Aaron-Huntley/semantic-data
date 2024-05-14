# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:10:25 2024

@author: ahuntle4
"""

import numpy as np
from analysis_tools import *
import math

from numpy.random import default_rng
rng = default_rng(42)  # Create a random number generator

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix

from gtda.graphs import GraphGeodesicDistance
from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, FlagserPersistence

from igraph import Graph

from IPython.display import SVG, display




data = pd.read_csv("Dataset.csv")

valence = [data.iloc[:, 0].values,data.iloc[:, 22].values]
V = [distance_matrix(valence)]
arousal = [data.iloc[:, 0].values,data.iloc[:, 23].values]
A = [distance_matrix(arousal)]

VR = VietorisRipsPersistence(metric="precomputed")

# Compute persistence diagrams corresponding to each entry (only one here) in X
diagramsV = VR.fit_transform(V)
diagramsA = VR.fit_transform(A)

VR.plot(diagramsV)
VR.plot(diagramsA)



print(f"diagrams.shape: {diagramsV.shape} ({diagramsV.shape[1]} topological features)")
print(f"diagrams.shape: {diagramsA.shape} ({diagramsA.shape[1]} topological features)")
    