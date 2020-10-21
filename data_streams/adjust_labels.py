'''
Created on 11 de out de 2018
@author: gusta
'''
import numpy as np

class Adjust_labels:
    def __init__(self):
        '''
        class to adjust the labels of the streams
        '''
        pass
    
    def targetStream(self, labels, data):
        '''
        metodo para ajustar a saida do stream, transformar dados categoricos em numeros
        :param: labels: rotulos existentes no stream
        :param: data: stream
        :return: stream corrigido
        '''
        
        data = np.asarray(data)
        
        # alterando os labels
        if(labels == ['1', '2', '3']):
            index = [0 if int(i) == 3 else int(i) for i in data[:, -1]]
        elif(labels == ['1', '2', '3', '4']):
            index = [0 if int(i) == 4 else int(i) for i in data[:, -1]]
        elif(labels == ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']):
            index = data[:, -1]
        elif(labels == ['0', '1']):
            index = data[:, -1]
        elif(labels == ['0', '1', '2']):
            index = data[:, -1]
        elif(labels == ['n', 'p'] or labels == ['p', 'n']):
            index = [0 if i == 'p' else 1 if i == 'n' else i for i in data[:, -1]]
        elif(labels == ['1', '2']):
            index = [0 if int(i) == 2 else int(i) for i in data[:, -1]]
        else:
            index = data[:, -1]
            
        index = np.asarray(index)
        return index.astype(int)
    
    def adjustStream(self, labels, data):
        '''
        metodo para ajustar a saida do stream, transformar dados categoricos em numeros
        :param: labels: rotulos existentes no stream
        :param: data: stream
        :return: stream corrigido
        '''
        
        # transforming into array
        data = np.asarray(data)
        
        # receiving the adjusted label
        index = self.targetStream(labels, data)
        # passing the new labels to the stream
        for i in range(len(index)):
            data[i, -1] = index[i]
        
        data = np.asarray(data, dtype='float')
    
        # returning the stream
        return data
     
    