'''
Created on 3 de out de 2018
@author: gusta
'''

import numpy as np

class VirtualStream():
    def __init__(self):
        '''
        Class to generate datasets with virtual concept drifts
        '''
    
        self.S = np.asarray([[0.3, 0.4, 0.6, 0.3],
                            [0.6, 0.4, 0.2, 0.7],
                            [0.4, 0.3, 0.7, 0.9],
                            [0.3, 0.8, 0.4, 0.5],
                            [0.6, 0.3, 0.9, 0.4]])
            
        self.STREAM = []
        self.NAME = []
        
    def gaussians_changing_variance(self, mu, y_new, concept, std):
        '''
        method to generate gaussians as dataset
        :param: mu: centroid to generate the dataset
        :param: y_new: to repeat the class of the 4th gaussian
        :param: concept: to choose the type of change
        :param: std: standard deviation of data
        :return: pattern [x, y] 
        '''
        
        if(len(mu) > 3):
            y = np.random.randint(4)
            x = np.random.gumbel((1, std))
            x = x * self.S[concept, y]
            x = x + mu[y]
            
            if(y == 3):
                y = y_new
            
            return np.append(x, y)
            
        else:
            y = np.random.randint(3)
            x = np.random.gumbel((1, std))
            x = x * self.S[concept, y]
            x = x + mu[y]
            
            return np.append(x, y)
        
    def generate_stream_5changes(self, concept, i):
        '''
        method to generate the stream
        :param: concept: type of concept [0...4]
        :param: lenght: size of stream
        :return: the stream
        '''
        
        self.NAME = 'virtual_5changes_' + str(i)
        
        for i in range(10000):
            
            if(i < 2000):
                mean = np.asarray([[1, 3], [1, 2], [2.5, 2.5]])
                x = self.gaussians_changing_variance(mean, 0, concept, 1)
                self.STREAM.append(x)
            elif(i > 2000 and i < 4000):
                mean = np.asarray([[1, 3], [1, 2], [2.5, 2.5], [3, 1]])
                x = self.gaussians_changing_variance(mean, 0, concept, 1)
                self.STREAM.append(x)
            elif(i > 4000 and i < 6000):
                mean = np.asarray([[3, 1], [1, 2], [2.5, 2.5]])
                x = self.gaussians_changing_variance(mean, 0, concept, 1)
                self.STREAM.append(x)
            elif(i > 6000 and i < 8000):
                mean = np.asarray([[3, 1], [1, 2], [2.5, 2.5], [5, 2]])
                x = self.gaussians_changing_variance(mean, 1, concept, 1)
                self.STREAM.append(x)
            elif(i > 8000):
                mean = np.asarray([[3, 1], [5, 2], [2.5, 2.5]])
                x = self.gaussians_changing_variance(mean, 1, concept, 1)
                self.STREAM.append(x)
                
                
        self.STREAM = np.asarray(self.STREAM)

    def generate_stream_9changes(self, concept, i):
        '''
        method to generate the stream
        :param: concept: type of concept [0...4]
        :param: lenght: size of stream
        :return: the stream
        '''
        
        self.NAME = 'virtual_9changes_' + str(i)
        
        for i in range(9000):
            
            #ok
            if(i < 1000):
                mean = np.asarray([[1, 3], [1, 2], [2.5, 2.5]])
                x = self.gaussians_changing_variance(mean, 0, concept, 1)
                self.STREAM.append(x)
                
            #ok
            elif(i > 1000 and i < 2000):
                mean = np.asarray([[1, 3], [1, 2], [2.5, 2.5], [3, 1]])
                x = self.gaussians_changing_variance(mean, 0, concept, 1)
                self.STREAM.append(x)
                
            #ok
            elif(i > 2000 and i < 3000):
                mean = np.asarray([[3, 1], [1, 2], [2.5, 2.5]])
                x = self.gaussians_changing_variance(mean, 0, concept, 1)
                self.STREAM.append(x)
                
            #ok
            elif(i > 3000 and i < 4000):
                mean = np.asarray([[3, 1], [1, 2], [2.5, 2.5], [5, 2]])
                x = self.gaussians_changing_variance(mean, 1, concept, 1)
                self.STREAM.append(x)
                
            #ok
            elif(i > 4000 and i < 5000):
                mean = np.asarray([[3, 1], [5, 2], [2.5, 2.5]])
                x = self.gaussians_changing_variance(mean, 1, concept, 1)
                self.STREAM.append(x)
            
            #ok
            elif(i > 5000 and i < 6000):
                mean = np.asarray([[3, 1], [5, 2], [2.5, 2.5], [0, -1]])
                x = self.gaussians_changing_variance(mean, 2, concept, 1)
                self.STREAM.append(x)
                
            #ok
            elif(i > 6000 and i < 7000):
                mean = np.asarray([[3, 1], [5, 2], [0, -1]])
                x = self.gaussians_changing_variance(mean, 2, concept, 1)
                self.STREAM.append(x)
                
            #ok
            elif(i > 7000 and i < 8000):
                mean = np.asarray([[3, 1], [5, 2], [0, -1], [1, 3]])
                x = self.gaussians_changing_variance(mean, 1, concept, 1)
                self.STREAM.append(x)
            
            #ok
            elif(i > 8000):
                mean = np.asarray([[3, 1], [1, 3], [0, -1]])
                x = self.gaussians_changing_variance(mean, 0, concept, 1)
                self.STREAM.append(x)
                
        self.STREAM = np.asarray(self.STREAM)
    
    def write_to_arff(self, output_path):
        '''
        method to write arff file
        :param: output_path: location to store the data
        '''
        
        arff_writer = open(output_path+self.NAME+".arff", "w")
        arff_writer.write("@relation VIRTUAL" + "\n")
        arff_writer.write("@attribute x real" + "\n" +
                          "@attribute y real" + "\n" +
                          "@attribute class {0,1,2}" + "\n\n")
        arff_writer.write("@data" + "\n")
        for i in range(0, len(self.STREAM)):
            arff_writer.write(str("%0.3f" % self.STREAM[i][0]) + "," +
                              str("%0.3f" % self.STREAM[i][1]) + "," +
                              str("%d" % self.STREAM[i][2]) + "\n")
        arff_writer.close()
        print("You can find the generated files in " + output_path + self.NAME + "!")
        
        
    



    
    