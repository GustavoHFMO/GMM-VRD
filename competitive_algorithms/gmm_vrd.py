'''
Created on 22 de ago de 2018
By Gustavo Oliveira
Universidade Federal de Pernambuco, Recife, Brasil
E-mail: ghfmo@cin.ufpe.br

OLIVEIRA, Gustavo HFM; MINKU, Leandro L.; OLIVEIRA, Adriano LI. 
GMM-VRD: A Gaussian Mixture Model for Dealing With Virtual and Real Concept Drifts. 
In: 2019 International Joint Conference on Neural Networks (IJCNN). 
IEEE, 2019. p. 1-8.
url:https://ieeexplore.ieee.org/abstract/document/8852097/
'''

from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
from sklearn.metrics import accuracy_score
from data_streams.adjust_labels import Adjust_labels
from streams.readers.arff_reader import ARFFReader
from gaussian_models.gmm_unsupervised import Gaussian
from gaussian_models.gmm_unsupervised import GMM_KDN
from detectors.ewma import EWMA
al = Adjust_labels()
import numpy as np
import time
np.random.seed(0)

class GMM_VD(GMM_KDN):
    def __init__(self):
        super().__init__()
        self.noise_threshold = 0.7
        self.n_vizinhos = 5

    '''
    METHOD INITIALIZATION
    '''
    
    def start(self, train_input, train_target, noise_threshold=False, n_vizinhos=False):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        :param: Kmax: number max of gaussians to test. Default 4
        :param: restarts: integer - number of restarts. Default 1
        :param: iterations: integer - number of iterations to trains the gmm model. Default 30
        '''
        
        # training the GMM
        self.fit(train_input, train_target, self.noise_threshold, self.n_vizinhos)
         
    '''
    VIRTUAL ADAPTATION
    '''
        
    def virtualAdaptation(self, x, y_true, y_pred):
        '''
        method to update an gaussian based on error 
        '''
        
        if(y_true != y_pred):
            
            # find the nearest gaussian
            flag, gaussian = self.nearestGaussian(x, y_true)
                    
            # condition
            if(flag):
                # update the nearest gaussian
                self.updateGaussian(x, gaussian)
                
            else:
                # create a gaussian
                self.createGaussian(x, y_true)
                
            # removing obsolete gaussians
            self.removeGaussians()
        
    '''
    ONLINE LEARNING
    '''
        
    def nearestGaussian(self, x, y):
        '''
        method to find the nearest gaussian of the observation x
        :x: observation 
        :y: label
        '''
        
        # receiving the gaussian with more probability for the pattern
        z = [0] * len(self.gaussians)
        for i in range(len(self.gaussians)):
            if(self.gaussians[i].label == y):
                z[i] = self.conditionalProbability(x, i)

        # nearest gaussian
        gaussian = np.argmax(z)
        
        # returning the probability and the nearest gaussian
        if(z[gaussian] > 0):
            return True, gaussian
        else:
            return False, gaussian 
        
    def updateGaussian(self, x, gaussian):
        '''
        method to update the nearest gaussian of x
        :x: the observation that will be used to update a gaussian
        :gaussian: the number of gaussian that will be updated  
        '''

        # updating the likelihood of all gaussians for x
        self.updateLikelihood(x)
        
        # updating the gaussian weights
        self.updateWeight()
                
        # storing the old mean
        old_mean = self.gaussians[gaussian].mu
        
        # updating the mean
        self.gaussians[gaussian].mu = self.updateMean(x, gaussian)

        # updating the covariance        
        self.gaussians[gaussian].sigma = self.updateCovariance(x, gaussian, old_mean)
        
    def updateLikelihood(self, x):
        '''
        method to update the parameter cver
        :param: x: new observation
        '''
        
        # getting the probabilities
        probabilities = self.posteriorProbabilities(x)
        
        # updating the loglikelihood
        for i in range(len(self.gaussians)):
            self.gaussians[i].dens += probabilities[i]
        
    def updateMean(self, x, gaussian):
        '''
        Method to update the mean of a gaussian i
        :x: pattern
        :gaussian: gaussian number
        return new mean
        '''
        
        # computing the new mean
        part1 = self.posteriorProbability(x, gaussian)/self.gaussians[gaussian].dens
        part2 = np.subtract(x, self.gaussians[gaussian].mu)
        new = self.gaussians[gaussian].mu + (np.dot(part1, part2))
        
        # returning mean
        return new
    
    def updateCovariance(self, x, i, old_mean):
        '''
        Method to update the mean of a gaussian i
        :x: pattern
        :gaussian: gaussian number
        return new mean
        '''
        
        # equation to compute the covariance
        ######## primeira parte ##############
        # sigma passado
        part0 = self.gaussians[i].sigma
        
        # primeiro termo
        part1 = np.subtract(self.gaussians[i].mu, old_mean)
        
        # segundo termo transposto
        part2 = np.transpose([part1])
        
        # multiplicacao dos termos
        part3 = np.dot(part2, [part1])
        
        # subtracao do termo pelo antigo
        part4 = np.subtract(part0, part3)
        ########################################
        
        
        ######## segunda parte ##############
        #ajuste de pertinencia
        part5 = self.posteriorProbability(x, i)/self.gaussians[i].dens
        
        # primeiro termo
        part6 = np.subtract(x, self.gaussians[i].mu)
        
        # segundo termo transposto
        part7 = np.transpose([part6])
        
        # multiplicacao do primeiro pelo segundo
        part8 = np.dot(part7, [part6])
        
        # subtracao do sigma antigo pelos termos
        part9 = np.subtract(part8, part0)
        
        
        # multiplicacao da pertinencia pelo colchetes
        part10 = np.dot(part5, part9)
        ########################################
        
        
        #final
        covariance = np.add(part4, part10) 
        
        # returning covariance
        return covariance

    '''
    CREATING GAUSSIANS ONLINE
    '''
    
    def createGaussian(self, x, y):
        '''
        method to create a new gaussian
        :x: observation 
        :y: label
        '''
        
        # mu
        mu = x
        
        # covariance
        cov = (0.05**2) * np.identity(len(x))
        
        # label
        label = y
        
        # new gaussian
        g = Gaussian(mu, cov, 1, label)
        
        # adding the new gaussian in the system
        self.gaussians.append(g)
        
        # adding 
        self.K += 1
        
        # updating the density of all gaussians
        self.updateLikelihood(x)
        
        # updating the weights of all gaussians
        self.updateWeight()
    
    def removeGaussians(self):
        '''
        method to remove obsolete gaussians
        '''
        
        # ammount of gaussians per class
        class_gaussians = [self.gaussians[i].label for i in range(len(self.gaussians))]
        labels, ammount = np.unique(class_gaussians, return_counts=True)
            
        # to search obsolete gaussians
        erase = []
        for i in range(len(labels)):
            for j in range(len(self.gaussians)):
                if(ammount[i] > 1 and self.gaussians[j].label == labels[i] and self.gaussians[j].mix < 0.01):
                    erase.append(j)
        
        # to remove obsolete gaussians
        for i in sorted(erase, reverse=True):
            del self.gaussians[i]
            self.K -= 1
            
class GMM_VRD(PREQUENTIAL_SUPER):
    def __init__(self, window_size=200):
        '''
        method to use an gmm with a single train to classify on datastream
        :param: classifier: class with the classifier that will be used on stream
        :param: stream: stream that will be tested
        '''

        # main variables
        self.CLASSIFIER = GMM_VD()
        self.DETECTOR = EWMA(min_instance=window_size, c=1, w=0.5)
        self.WINDOW_SIZE = window_size
        self.LOSS_STREAM = []
        self.DETECTIONS = [0]
        self.WARNINGS = [0]
        self.CLASSIFIER_READY = True
        
        # auxiliar variables
        self.NAME = 'GMM-VRD'
        self.TARGET = []
        self.PREDICTIONS = []
        self.count = 0
    
    '''
    WINDOW MANAGEMENT
    '''
        
    def transferKnowledgeWindow(self, W, W_warning):    
        '''
        method to transfer the patterns of one windown to other
        :param: W: window that will be updated
        :param: W_warning: window with the patterns that will be passed
        '''
        
        W = W_warning
        
        return W
    
    def manageWindowWarning(self, W, x):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        if(self.CLASSIFIER_READY):
            W = self.incrementWindow(W, x)
            
            if(len(W) > self.WINDOW_SIZE/2):
                W = self.resetWindow(W)
        
        return W
     
    def resetWindow(self, W):
        '''
        method to reset the window
        :param: W: window that will be updated 
        '''
        
        return np.array([])
    
    def incrementWindow(self, W, x):
        '''
        method to icrement the window under the example
        :param: W: window that will be updated 
        :param: x: example that will be inserted 
        '''
        
        aux = [None] * (len(W)+1)
        aux[:-1] = W
        aux[-1] = x
        
        return np.asarray(aux) 
    
    def slidingWindow(self, W, x):
        '''
        method to slide the window under the example
        :param: W: window that will be updated 
        :param: x: example that will be inserted 
        '''
        
        aux = [None] * len(W)
        aux[0:-1] = W[1:]
        aux[-1] = x
    
        return np.asarray(aux)

    '''
    FIT THE CLASSIFIER
    '''
    
    def trainClassifier(self, W):
        '''
        method to train an classifier on the data into W
        :param: W: set with the patterns that will be trained
        '''
        
        # split patterns and labels
        x_train, y_train = W[:,0:-1], W[:,-1]
        
        # fitting the dataset
        self.CLASSIFIER.fit(x_train, y_train)

        # printing the accuracy for training set
        pred = self.CLASSIFIER.predict(x_train)
        print(accuracy_score(y_train, pred))
        
        # returning the new classifier        
        return self.CLASSIFIER
    
    '''
    RUN ON DATA STREAM
    '''
    
    def run(self, labels, stream, cross_validation=False, fold=5, qtd_folds=30):
        '''
        method to run the stream
        '''
        
        # starting time
        self.start_time = time.time()
        
        ######################### 1. FITTING THE STREAM AND AUXILIAR VARIABLES ####################
        # storing the new stream
        self.STREAM = stream
        
        # obtaining the initial window
        W = self.STREAM[:self.WINDOW_SIZE]
        
        # instantiating the validation window
        W_validation = W
        ######################### 1. #############################################################
        
         
         
         
        ########################### 2. STARTING THE CLASSIFIER AND DETECTOR #########################
        # training the classifier
        self.CLASSIFIER = self.trainClassifier(W)
        
        # fitting the detector
        self.DETECTOR.fit(self.CLASSIFIER, W) 
        
        # instantiating a window for warning levels
        W = [] 
        W_warning = []
        ############################ 2. ##############################################################
        
        
        #################################### 3.SIMULATING THE STREAM ################################
        # for to operate into a stream
        for i, X in enumerate(self.STREAM[self.WINDOW_SIZE:]):
            
            # to use the cross validation
            run=False
            if(cross_validation and self.cross_validation(i, qtd_folds, fold)):
                run = True
            
            # to execute the prequential precedure
            if(run):
                # split the current example on pattern and label
                x, y = X[0:-1], int(X[-1])
        ##################################### 3. ######################################################
        
        
        
        
                ########################################## 4. ONLINE CLASSIFICATION ###################################
                # using the classifier to predict the class of current label
                yi = self.CLASSIFIER.predict(x)
                
                # storing the predictions
                self.PREDICTIONS.append(yi)
                self.TARGET.append(y)
                ########################################## 4. #########################################################
                
                
                
                
                
                ########################################## 5. VIRTUAL ADAPTATION #######################################
                # sliding the current observation into W
                W_validation = self.slidingWindow(W_validation, X)
                    
                # updating the gaussian if the classifier miss
                self.CLASSIFIER.virtualAdaptation(x, y, yi)
                ######################################### 5. ###########################################################
                
                
                
                
                ################################ 7. MONITORING THE DRIFT  ##############################################
                # verifying the claassifier
                if(self.CLASSIFIER_READY):
    
                    # monitoring the datastream
                    warning_level, change_level = self.DETECTOR.detect(y, yi)
                ################################## 7. ####################################################################
                
                            
                    
                    
                    ################################## 8. WARNING ERROR PROCEDURES ###########################################
                    if(warning_level):
    
                        # managing the window warning
                        W_warning = self.manageWindowWarning(W_warning, X)
                        
                        # storing the time when warning was triggered
                        self.WARNINGS.append(i)
                    ################################## 8. ####################################################################
                    
                    
                        
                
                    ################################## 9. DRIFT ERROR PROCEDURES ############################################
                    if(change_level):
                        
                        # storing the time of change
                        self.DETECTIONS.append(i)
                        
                        # reseting the detector
                        self.DETECTOR.reset()
                            
                        # reseting the window
                        W = self.transferKnowledgeWindow(W, W_warning)
                        
                        # reseting the classifier 
                        self.CLASSIFIER_READY = False
                    ################################## 9. ####################################################################
                    
                    
                
                
                ################################## 10. COLLECTING NEW DATA ############################################
                elif(self.WINDOW_SIZE > len(W)):
                    
                    # sliding the current observation into W
                    W = self.incrementWindow(W, X)
                ################################## 10. ################################################################
                
                
                
                
                ################################## 11. RETRAINING THE CLASSIFIER #########################################
                else:
                    
                    # to remodel the knowledge of the classifier
                    self.CLASSIFIER = self.trainClassifier(W)
                        
                    # fitting the detector
                    self.DETECTOR.fit(self.CLASSIFIER, W) 
                            
                    # releasing the new classifier
                    self.CLASSIFIER_READY = True
                ################################## 11. ###################################################################
                
                    
                # print the current process
                self.printIterative(i)
                
        # ending time
        self.end_time = time.time()
    
def main():
    
    
    ####### 1. DEFINING THE DATASETS ##################################################################
    i = 4
    # REAL DATASETS 
    dataset = ['PAKDD', 'elec', 'noaa', 'poker-lsn2', 'covtypeNorm2']
    labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    stream_records = al.adjustStream(labels, stream_records)
    
    # SYNTHETIC DATASETS 
    #dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    #labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    ####### 1. ########################################################################################
    
    
    ####### 2. DEFINING THE MODEL PARAMETERS ##########################################################
    # instantiate the prequetial
    preq = GMM_VRD(window_size=200)
    preq.run(labels, stream_records, cross_validation=True, qtd_folds=30, fold=1)
    
    # presenting the accuracy
    preq.plotAccuracy()
    print("Accuracy: ", preq.accuracyGeneral())
    ####### 2. DEFINING THE MODEL PARAMETERS ##########################################################
    
    
    ####### 3. STORING THE PREDICTIONS ################################################################
    import pandas as pd
    df = pd.DataFrame(data={'predictions': preq.PREDICTIONS, 'target':preq.TARGET})
    df.to_csv("../projects/"+preq.NAME+"-"+dataset[i]+".csv")
    ####### 3. STORING THE PREDICTIONS ################################################################
    
if __name__ == "__main__":
    main()        
           
        
    