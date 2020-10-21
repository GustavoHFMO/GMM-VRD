'''
Created on 11 de out de 2018
@author: gusta
'''

import numpy as np

class PrototypeSelection:
    def __init__(self):
        pass

    def BIC(self, L, p, n):
        '''
        method for calculate the bayesian information criterion
        :L: value of likelihood function
        :p: number of parameters of the model
        :n: number of observations
        :return: bic value for the respective parameters
        '''
        return np.log(n) * p - 2 * L  
    
    def AIC(self, L, p):
        '''
        method for calculate the akaike information criterion
        :L: value of likelihood function
        :p: number of parameters of the model
        :return: bic value for the respective parameters
        '''
        return 2 * p - 2 * L    
    
    def prototype_metric(self, name, L, p, n):
        '''
        method to select the prototype selection metric
        :name: name of metric [BIC, AIC]
        :L: value of likelihood function
        :p: number of parameters of the model
        :n: number of observations
        :return: value for the respective parameters
        '''
        
        self.NAME = name
        
        if(self.NAME == 'BIC'):
            return self.BIC(L, p, n)
        elif(self.NAME == 'AIC'):
            return self.AIC(L, p)
        
        
        
        
        
        
    