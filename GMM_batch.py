'''
Created on 3 de mar de 2021
@author: gusta
'''

# IMPORTING THE ALGORITHMS
from competitive_algorithms.gmm_vrd import GMM_KDN
# Importing some libs to handle the dataset
from streams.readers.arff_reader import ARFFReader
from data_streams.adjust_labels import Adjust_labels
al = Adjust_labels()
# Libs to evaluate the offline model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

####### 1. DEFINING THE DATASETS ##################################################################
i = 4
# SYNTHETIC DATASETS 
dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes']
labels, _, stream_records = ARFFReader.read("data_streams/_synthetic/"+dataset[i]+".arff")
stream_records = al.adjustStream(labels, stream_records[400:1000])

# splitting the dataset into (70%) for training and (30%) for test
X_train, X_test, y_train, y_test = train_test_split(stream_records[:,0:-1], stream_records[:,-1], test_size=0.3, random_state=0)
####### 1. ########################################################################################
    
    
####### 2. TRAINING GMM ###########################################################################
# training the gmm model
gmm = GMM_KDN(noise_threshold=0.8, n_vizinhos=7, kmax=2, emit=10)
gmm.fit(X_train, y_train)

# plotting the model trained
training_accuracy = accuracy_score(y_train, gmm.predict(X_train))
gmm.train_input = X_train
gmm.train_target = y_train
gmm.plotGmm("GMM in the training set", accuracy=training_accuracy)

# evaluating and plotting the model in the test set
test_accuracy = accuracy_score(y_test, gmm.predict(X_test))
gmm.train_input = X_test
gmm.train_target = y_test
gmm.plotGmm("GMM in the test set", accuracy=test_accuracy)
###################################################################################################
