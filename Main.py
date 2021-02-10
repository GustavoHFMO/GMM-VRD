#-*- coding: utf-8 -*-

'''
Created on 22 de ago de 2018
By Gustavo Oliveira
Universidade Federal de Pernambuco, Recife, Brasil
E-mail: ghfmo@cin.ufpe.br

ALGORITHMS USED IN THE PAPER PUBLISHED BELOW:

OLIVEIRA, Gustavo HFM; MINKU, Leandro L.; OLIVEIRA, Adriano LI. 
GMM-VRD: A Gaussian Mixture Model for Dealing With Virtual and Real Concept Drifts. 
In: 2019 International Joint Conference on Neural Networks (IJCNN). 
IEEE, 2019. p. 1-8.
url:https://ieeexplore.ieee.org/abstract/document/8852097/
'''

# IMPORTING THE ALGORITHMS
from competitive_algorithms.gmm_vrd import GMM_VRD
from competitive_algorithms.dynse import Dynse
from competitive_algorithms.igmmcd import IGMM_CD

# Importing some libs to help the execution
from streams.readers.arff_reader import ARFFReader


####### 1. DEFINING THE DATASETS ##################################################################
i = 2
# REAL DATASETS 
#dataset = ['PAKDD', 'elec', 'noaa']
#labels, _, stream_records = ARFFReader.read("data_streams/real/"+dataset[i]+".arff")
    
# SYNTHETIC DATASETS 
dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/"+dataset[i]+".arff")
####### 1. ########################################################################################
    
    
####### 2. DEFINING THE MODELS ####################################################################
# GMM-VRD
gmm_vrd = GMM_VRD()
gmm_vrd.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
# presenting the accuracy
gmm_vrd.plotAccuracy()
print("Accuracy: ", gmm_vrd.accuracyGeneral())

# Dynse
dynse = Dynse()
dynse.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
# presenting the accuracy
dynse.plotAccuracy()
print("Accuracy: ", dynse.accuracyGeneral())

# IGMM-CD
igmm_cd = IGMM_CD()
igmm_cd.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
# presenting the accuracy
igmm_cd.plotAccuracy()
print("Accuracy: ", igmm_cd.accuracyGeneral())
####### 2. DEFINING THE MODELS ####################################################################
    
    
####### 3. STORING THE PREDICTIONS ################################################################
import pandas as pd
df = pd.DataFrame(data={'predictions': gmm_vrd.PREDICTIONS, 'target':gmm_vrd.TARGET})
df.to_csv("../projects/"+gmm_vrd.NAME+"-"+dataset[i]+".csv")
####### 3. STORING THE PREDICTIONS ################################################################