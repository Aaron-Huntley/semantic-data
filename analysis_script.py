# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:41:02 2024

@author: ahuntle4
"""

from analysis_tools import *
import math

def import_all_linear(F=True):
    '''produces jaccard graphs for each pair of data in the dataset.
    Linerarly normalised'''
    
    data = pd.read_csv("Dataset.csv")
    
    for i in range(1, 55):
        
          tempraw1 = [data.iloc[:, 0].values,data.iloc[:, i].values]
          filtered_data = [x for x in tempraw1[1] if not isinstance(x, float) or not math.isnan(x)]
          tempraw1N = normalise_linear(tempraw1, math.floor(min(filtered_data)),math.ceil(max(filtered_data)) , 0, 8,F)
         
          for j in range(i,55):
             
              tempraw2 = [data.iloc[:, 0].values,data.iloc[:, j].values]
              filtered_data2 = [x for x in tempraw2[1] if not isinstance(x, float) or not math.isnan(x)]
              tempraw2N = normalise_linear(tempraw2, math.floor(min(filtered_data2)), math.ceil(max(filtered_data2)), 0, 8,F)
              if F ==True:
                  percent_compare_linear(tempraw1N,data.columns[i],tempraw2N,data.columns[j],True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/Graphs')
              elif F==False:
                  percent_compare_linear(tempraw1N,data.columns[i],tempraw2N,data.columns[j],True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/Graphs/inverse')
                  
def import_all_ordered():
    '''produces jaccard graphs for each pair of data in the dataset.
      normalised by order'''
      
    data = pd.read_csv("Dataset.csv")
      
    for i in range(1,55):
        tempraw1 = [data.iloc[:, 0].values, data.iloc[:, i].values]
        raw1N = normalise_sort(tempraw1)
      
        for j in range(i, 55):
            tempraw2 = [data.iloc[:, 0].values, data.iloc[:, j].values]
            raw2N = normalise_sort(tempraw2)  # Fixed the variable name here
                
            percent_compare_ordered(raw1N, data.columns[i], raw2N, data.columns[j], True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/Graphs')

def compare_interesting():
    '''Produces ordered Jaccard graphs of intresting pairs'''
    
    data = pd.read_csv("Dataset.csv")
    
    valence_warriner = [data.iloc[:, 0].values,data.iloc[:, 22].values]
    arousal_warriner = [data.iloc[:, 0].values,data.iloc[:, 23].values]
    valence_mohammad = [data.iloc[:, 0].values,data.iloc[:, 28].values]
    arousal_mohammad = [data.iloc[:, 0].values,data.iloc[:, 29].values]
    socialness = [data.iloc[:, 0].values,data.iloc[:, 1].values]
    interoception_lanc = [data.iloc[:, 0].values,data.iloc[:, 12].values]
    
    emotion_warriner_min = combine_raw(valence_warriner, arousal_warriner, "min")
    emotion_mohammad_min = combine_raw(valence_mohammad, arousal_mohammad, "min")
    emotion_warriner_ave = combine_raw(valence_warriner, arousal_warriner, "mean")
    emotion_mohammad_ave = combine_raw(valence_mohammad, arousal_mohammad, "mean")
    
    valence_warriner = normalise_sort(valence_warriner)
    arousal_warriner = normalise_sort(arousal_warriner)
    
    valence_mohammad = normalise_sort(valence_mohammad)
    arousal_mohammad = normalise_sort(arousal_mohammad)
    
    socialness = normalise_sort(socialness)
    interoception_lanc = normalise_sort(interoception_lanc)
    
    emotion_warriner_min = normalise_sort(emotion_warriner_min)
    emotion_mohammad_min = normalise_sort(emotion_mohammad_min)
    
    emotion_warriner_ave = normalise_sort(emotion_warriner_ave)
    emotion_mohammad_ave = normalise_sort(emotion_mohammad_ave)
    
    
    percent_compare_ordered(valence_warriner, 'valence_warriner', valence_mohammad, 'valence_mohammad', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    percent_compare_ordered(arousal_warriner, 'arousal_warriner', arousal_mohammad, 'arousal_mohammad', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    
    percent_compare_ordered(emotion_warriner_min, 'emotion_warriner_min', emotion_mohammad_min, 'emotion_mohammad_min', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    percent_compare_ordered(emotion_warriner_ave, 'emotion_warriner_ave', emotion_mohammad_ave, 'emotion_mohammad_ave', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    
    percent_compare_ordered(interoception_lanc, 'interoception_lanc', valence_warriner, 'valence_warriner', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    percent_compare_ordered(interoception_lanc, 'interoception_lanc', valence_mohammad, 'valence_mohammad', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    percent_compare_ordered(interoception_lanc, 'interoception_lanc', arousal_warriner, 'arousal_warriner', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    percent_compare_ordered(interoception_lanc, 'interoception_lanc', arousal_mohammad, 'arousal_mohammad', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
   
    percent_compare_ordered(interoception_lanc, 'interoception_lanc', arousal_warriner, 'emotion_warriner_min', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    percent_compare_ordered(interoception_lanc, 'interoception_lanc', arousal_mohammad, 'emotion_mohammad_min', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    percent_compare_ordered(interoception_lanc, 'interoception_lanc', arousal_warriner, 'emotion_warriner_ave', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    percent_compare_ordered(interoception_lanc, 'interoception_lanc', arousal_mohammad, 'emotion_mohammad_ave', True,'C:/Users/ahuntle4/.spyder-py3/Data Anal/CoolGraphs')
    
    
    
'''TEST RUNNING'''

compare_interesting()

# import_all_ordered()
# import_all_linear()
# import_all_linear(False)






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
