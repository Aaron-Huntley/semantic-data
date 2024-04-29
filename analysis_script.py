# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:41:02 2024

@author: ahuntle4
"""

from analysis_tools import *
import math

def import_all_linear():
    '''produces jaccard graphs for each pair of data in the dataset.
    Linerarly normalised'''
    
    data = pd.read_csv("Dataset.csv")
    
    for i in range(1, 10):
        
          tempraw1 = [data.iloc[:, 0].values,data.iloc[:, i].values]
          tempraw1N = normalise_linear(tempraw1, math.floor(min(tempraw1[1])),math.ceil(max(tempraw1[1])) , 0, 8)
         
          for j in range(i,10):
             
              tempraw2 = [data.iloc[:, 0].values,data.iloc[:, j].values]
              tempraw2N = normalise_linear(tempraw2, math.floor(min(tempraw2[1])), math.ceil(max(tempraw2[1])), 0, 8)
             
              percent_compare_linear(tempraw1N,data.columns[i],tempraw2N,data.columns[j],True)



def import_all_ordered():
    '''produces jaccard graphs for each pair of data in the dataset.
      normalised by order'''
      
    data = pd.read_csv("Dataset.csv")
      
    for i in range(1, 55):
        tempraw1 = [data.iloc[:, 0].values, data.iloc[:, i].values]
        raw1N = normalise_sort(tempraw1)
      
        for j in range(i, 55):
            tempraw2 = [data.iloc[:, 0].values, data.iloc[:, j].values]
            raw2N = normalise_sort(tempraw2)  # Fixed the variable name here
                
            percent_compare_ordered(raw1N, data.columns[i], raw2N, data.columns[j], True)





'''TEST RUNNING'''


#import_all_linear()

import_all_ordered()


'''Import Data'''

# valence = import_grounding(data,"Valence_Warriner")
# arousal = import_grounding(data,"Arousal_Warriner")
# social = import_grounding(data,"Socialness")
# intero = import_grounding(data,"Interoceptive")

# infinite =[[],[]]
# infinite[0] = social[0]
# infinite[1] = np.zeros(len(social[0]))

# valenceT = [[],[]]
# valenceT[0] = valence[0][:10]
# valenceT[1] = valence[1][:10]

# arousalT = [[],[]]
# arousalT[0] = arousal[0][:10]
# arousalT[1] = arousal[1][:10]

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

# Val_Aro = percent_compare_linear(valence,"Valence",arousal,"Arousal",True)
# Val_Soc = percent_compare_linear(valence,"Valence",social,"Social",True)

# Aro_Soc = percent_compare_linear(arousal,"Arousal",social,"Social",True)
# Aro_Int = percent_compare_linear(arousal,"Arousal",intero,"Intero",True)

# Val_Int = percent_compare_linear(valence,"Valence",intero,"Intero",True)
# Soc_Int = percent_compare_linear(social,"Social",intero,"Intero",True)

# Aro_Aro = percent_compare_linear(arousal,"Arousal",arousal,"Arousal",True)

'''Test combined'''

# emotion = combine_raw(valence, arousal, "min")

# # Emo_Int = percent_compare_linear(emotion,"Emotion",intero,"Intero",True)

# Inf_Int =  percent_compare_linear(infinite,"Inf",intero,"Intero",True)