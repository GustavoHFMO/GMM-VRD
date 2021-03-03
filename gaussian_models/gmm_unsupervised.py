'''
Created on 29 de abr de 2018
@author: gusta
'''

from gaussian_models.prototype_selection import PrototypeSelection
ps = PrototypeSelection()
from streams.readers.arff_reader import ARFFReader
import matplotlib.patches as patches
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import copy

class PlotGMM():
    '''
    class with methods to plot different graphs for GMM
    '''
    plt.style.use('seaborn-whitegrid')
    pass

    def plotGmm(self, t, accuracy=False, show=True, salvar=False):
        
        if(salvar):
            plt.clf()
        
        # defining the colors of each gaussian
        colors = cm.rainbow(np.linspace(0, 1, len(self.unique)))
        marks = ["^", "o", '+', ',']
        
        # creating the image
        plt.subplot(111)
            
        # receiving each observation per class
        classes = []
        for i in self.unique:
            aux = []
            for j in range(len(self.train_target)):
                if(self.train_target[j] == i):
                    aux.append(self.train_input[j])
            classes.append(np.asarray(aux))
        classes = np.asarray(classes)
        
        # plotting each class
        for i in range(len(self.unique)):
            plt.scatter(classes[i][:,0],
                        classes[i][:,1],
                        color = colors[i],
                        marker = marks[i],
                        label = 'class '+str(i)) 
            
        # plotting the gaussians under the dataset
        for i in range(len(self.gaussians)):
            c = colors[int(self.gaussians[i].label)]

            # plotting the number of gaussian
            plt.text(self.gaussians[i].mu[0], self.gaussians[i].mu[1], "G"+str(i), weight='bold')
                                
            # plotting the gaussian
            self.draw_ellipse(self.gaussians[i].mu, self.gaussians[i].sigma, c)
        
        
        if(accuracy != False): 
            accuracy = 100*accuracy
            texto = ("Accuracy: %.2f" % (accuracy))
    
            plt.annotate(texto,
                    xy=(0.5, 0.15), xytext=(0, 0),
                        xycoords=('axes fraction', 'figure fraction'),
                        textcoords='offset points',
                        size=16, ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))
        
        # definindo o titulo e mostrando a imagem
        #plt.title('Time: ' +str(t))
        plt.title(str(t))
        plt.legend(loc='upper right')
        
        # ajustando o layout
        plt.gcf().set_size_inches(4.5, 4)
        plt.subplots_adjust(bottom=0.08, top=0.92, left=0.08, right=0.99)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        
        if(salvar):
            plt.savefig("../projects/val/"+t+".png")
            
        if(show):
            plt.show()

    def draw_ellipse(self, position, covariance, color, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, _ = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)
        
        # Draw the Ellipse
        for nsig in range(2, 3):
            ax.add_patch(patches.Ellipse(position, 
                                         nsig * width, 
                                         nsig * height,
                                         angle, 
                                         #fill=False,
                                         color = color,
                                         linewidth=3,
                                         alpha=0.3, 
                                         **kwargs))
            
    def animation(self, it):
        '''
        method to call an animation
        :param: it: quantity of iterations necessary to simule 
        '''
        # creating the figure that will plot the gmm
        fig = plt.figure()
        img0 = fig.add_subplot(1, 2, 2)
        img1 = fig.add_subplot(1, 2, 1)
    
        # defining the colors of each gaussian
        colors = cm.rainbow(np.linspace(0, 1, len(self.gaussians)))
        
        # variable to store the evolution of loglikelihood
        self.listLoglike = []
        
        def update(i):
            '''
            method to call one plot
            '''
             
            print("[", i, "]")

            # erasing the img to plot a new figure
            img0.clear()
            img1.clear()
            
            #plotting the metrics 
            img1.plot(self.listLoglike, label='loglikelihood', color = 'r')
            img1.legend()
            
            #ploting the points
            img0.set_title(str('GMM - it: %d' % i))
            # creating the colours of each point
            indexColors = [np.argmax(self.matrixWeights[j]) for j in range(len(self.matrixWeights))]
            # plotting the dataset
            img0.scatter(self.train_input[:,0], self.train_input[:,1], c=colors[indexColors], label = 'dataset')
            # plotting the gaussians under the dataset
            for j in range(len(self.gaussians)):
                self.draw_ellipse(self.gaussians[j].mu, self.gaussians[j].sigma, colors[j], img0)
            
            # training the mixture model
            self.Mstep(self.Estep())
            self.listLoglike.append(np.abs(self.loglike))
            
        # function that update the animation
        _ = anim.FuncAnimation(fig, update, frames=it, repeat=False)
        plt.show()

class Gaussian:
    def __init__(self, mu, sigma, mix, label=None):
        '''
        Constructor of the Gaussian distribution
        :param: mu: the average of the data
        :param: sigma: the standard deviation of the data
        '''
        self.mu = mu
        self.sigma = sigma
        self.mix = mix
        self.label = label
        self.dens = 1
        
    def pdf_vector(self, x):
        '''
        Method to compute the probability of an vector
        :param: x: variable x which will be computed
        :return: the probability of the variable x belongs to this gaussian distribution
        '''
        
        # to avoid problems
        x = [0.01 if i == 0 else i for i in x]
        
        # adjusting the pattern
        x = np.transpose(np.array([x]))
        mu = np.transpose(np.array([self.mu]))
        
        # part 1
        # dealing with possible divisions by zero
        part1 = 1/(np.power(2*np.pi, len(x)/2) * np.sqrt(np.linalg.det(self.sigma)))
        if(part1 == np.float('inf')): part1 = 0.01
        
        # part 2
        part2 = np.transpose(np.subtract(x, mu))
        
        # part 3
        # dealing with zero on matrix
        try:
            part3 = np.linalg.inv(self.sigma)
        except:
            part3 = np.linalg.pinv(self.sigma)
        
        # part 4
        part4 = np.subtract(x, mu)
        
        # calculation of matrices
        a = np.dot(part2, part3)
        b = np.dot(a, part4)
        b = -0.5 * b[0][0]
        
        # an way to avoid problems with large values in b, because it result on infinity results
        c = np.exp(b)
        
        # final
        pertinence = part1 * c
        
        if(np.isnan(pertinence)):
            pertinence = 0.0

        # pertinence
        return pertinence
        
    def printstats(self):
        '''
        method to print the current mu and sigma of the distribution
        '''
        print('Gaussian: mi = {:.2}, sigma = {:.2}'.format(self.mu, self.sigma))
        
class GMM(PlotGMM):
    def __init__(self):
        pass

    
    '''
    UNSUPERVISED LEARN
    '''

    def fitClustering(self, train_input, K):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: K: integer - the quantity of Gaussians used
        '''
        
        # storing the dataset
        self.train_input = train_input
            
        # number of gaussians
        self.K = K
    
        # dividing the number of examples for each gaussian
        self.N = int(np.round(0.3 * len(train_input)))
                         
        # creating the gaussians
        self.gaussians = []
            
        # allocating itens for each gaussian
        for _ in range(K):
            # extracting random examples to calculate the first parameters
            if(len(self.train_input) > 5):
                randomData = [self.train_input[np.random.randint(1, len(self.train_input)-1)] for _ in range(self.N)]
            else: 
                randomData = self.train_input
    
            # creating the variables randomly
            g = Gaussian(np.mean(randomData, axis=0), np.cov(np.transpose(randomData)), 1./K)
                
            # storing the gaussians
            self.gaussians.append(g)
                
        # defining the quantity of parameters of model
        self.n = len(self.train_input)
        
        # quantity of means, covariances and constants of mixtures
        self.p = self.K * 3
            
        # intializing the matrix of weights
        self.matrixWeights = self.Estep()
    
    def trainEM(self, iterations, criterion=False):
        '''
        method to train the gaussians
        :param: iterations: integer - quantity of iterations necessary to train the models
        :param: log: boolean - variable to show the log of train
        '''
        
        # process to train the gmm
        self.listLoglike = []
        for i in range(iterations):
                
            # EM process
            self.Mstep(self.matrixWeights)
            self.matrixWeights = self.Estep()
            self.listLoglike.append(np.abs(self.loglike))
            
            if(i > 5):
                calculo = 100-(self.listLoglike[i]*100/self.listLoglike[i-1])
                if(calculo < 0.1):
                    #print("stop: ", i)
                    break
    
    def Estep(self):
        '''
        Method to compute the estimation of probability for each data
        :return: a matrix containing the weights for each data for all clusters 
        '''
        
        # (p=1)
        self.loglike = 0
        
        # matrix that will storage the weights
        matrixWeights = []
        
        # for to iterate for each data point
        for x in self.train_input:
            # for to estimate each weight of gaussians
            weights = [0] * self.K
            for i in range(self.K):
                # probability of instance x given the gaussian[i], this multiplied by the prior probability mix[i]
                weights[i] = self.conditionalProbability(x, i)
            
            # to avoid nan
            weights = np.nan_to_num(weights)
            
            # add into loglike
            self.loglike += np.log(np.sum(weights))
            
            # sum of probabilities, in other words, the sum of instance x belongs to all clusters, on the final is equal to one
            den = np.sum(weights)
            
            # completing the theorem of bayes for all clusters 
            weights /= den
            
            # returning the weights of all instances for all clusters
            matrixWeights.append(weights)
            
        # returning the weights
        return np.asarray(matrixWeights)
    
    def Mstep(self, matrixW):
        '''
        method to maximize the probabilities of the gaussians
        '''

        # updating each gaussian
        for i in range(self.K):
            
            # dividing the probabilities to each cluster
            wgts = matrixW[:,i]
            wgts = np.nan_to_num(wgts)
            
            # this variable is going to responsible to store the sum of probabilities for each cluster
            dens = np.sum(wgts)
            if(dens == 0):
                dens = 0.01
            
            # compute new means
            self.gaussians[i].mu = np.sum(prob*inst/dens for(prob, inst) in zip(wgts, self.train_input))

            # function to compute the covariance
            def covProb(mu, wgts, dens):
                '''
                submethod to update the covariance
                '''
                mu = np.transpose([mu])
                cvFinal = 0
                for i in range(len(wgts)):
                    dt = np.transpose([self.train_input[i]])
                    cv = np.dot(np.subtract(mu, dt), np.transpose(np.subtract(mu, dt)))
                    cv = wgts[i]*cv/dens
                    if(i==0):
                        cvFinal = cv
                    else:
                        cvFinal = np.add(cvFinal, cv)
                return cvFinal
                
            # compute new sigma (covariance)
            self.gaussians[i].sigma = covProb(self.gaussians[i].mu, wgts, dens)
                
            # compute new mix
            self.gaussians[i].dens = dens
            self.gaussians[i].mix = dens/len(self.train_input) 
    
    def chooseBestModel(self, train_input, type_selection, Kmax, restarts, iterations, stop_criterion=False):
        '''
        methodo to train several gmms and return the gmm with the best loglike
        :param: train_input: data that will be used to train the model
        :param: type_selection: name of prototype selection metric
        :param: Kmax: number max of gaussians to test
        :param: restarts: integer - number of restarts
        :param: iterations: integer - number of iterations to trains the gmm model
        :return: the best gmm model
        '''
        
        # creating the first gaussian
        gmm = copy.deepcopy(self)
        gmm.fitClustering(train_input, 1)
        gmm.trainEM(2)
        
        # evaluating the model
        bestMetric = ps.prototype_metric(type_selection, gmm.loglike, gmm.p, gmm.n)
        bestGmm = gmm
        
        # for to test several values of K
        for k in range(2, Kmax+1):
            
            # for to restart the models
            for _ in range(restarts):
                #print('K[', k, '] restart[', i, ']')
                gmm = copy.deepcopy(self)
                gmm.fitClustering(train_input, k)
                gmm.trainEM(iterations, stop_criterion)
                
                # evaluating the model
                metric = ps.prototype_metric(type_selection, gmm.loglike, gmm.p, gmm.n)
                
                # condition to store the best model
                if(metric < bestMetric):
                    bestMetric = metric
                    bestGmm = gmm
            
        # the best model
        return bestGmm
    
    
    '''
    PROBABILITY CALCULATION
    '''
    
    def posteriorProbability(self, x, i):
        '''
        method to return the posterior probability of an variable x to a gaussian i
        :param: x: observation
        :param: i: number of the gaussian
        '''
    
        # calculating the density
        dens = []
        for j in range(len(self.gaussians)):
            dens.append(self.conditionalProbability(x, j))
        
        # to avoid nan
        dens = np.nan_to_num(dens) 
        dens = np.sum(dens) 
        if(dens == 0.0): dens = 0.01  
        
        # posterior probability
        posterior = (self.conditionalProbability(x, i))/dens 
        
        # returning the posterior probability
        return posterior
    
    def conditionalProbability(self, x, i):
        '''
        method to return the conditional probability of an variable x to a gaussian i
        :param: x: observation
        :param: i: number of the gaussian
        '''
        
        # returning the conditional probability
        return self.gaussians[i].pdf_vector(x) * self.gaussians[i].mix
    
    def predictionProb(self, x):
        '''
        method to calculate the probability of a variable x to be on the distribution created
        :param: x: float - variable that we need to know the probability
        :return: the probability of the given variable
        '''
        
        # returning the pertinence for the observation
        pertinence = 0
        for i in range(len(self.gaussians)):
            pertinence += self.conditionalProbability(x, i)
        return pertinence
    
    
    '''
    CLASSIFICATION CALCULATION
    '''

    def predict(self, x):
        '''
        method to predict the class for several patterns
        :param: x: pattern
        :return: the respective label for x
        '''

        # to predict multiple examples
        if(len(x.shape) > 1):
            # returning all labels
            return [self.predict_one(pattern) for pattern in x]
        # to predict only one example
        else:
            return self.predict_one(x)
    
    def predict_one(self, x):
        '''
        method to predict the class for a only pattern x
        :param: x: pattern
        :return: the respective label for x
        '''
        
        # receiving the gaussian with more probability for the pattern
        y = [0]*len(self.gaussians)
        for i in range(len(self.gaussians)):
            y[i] = self.posteriorProbability(x, i)
        gaussian = np.argmax(y)    
                    
        # returning the label
        return self.gaussians[gaussian].label
    
    def predict_gaussian(self, x):
        '''
        method to predict the class for a only pattern x
        :param: x: pattern
        :return: the respective label for x
        '''
        
        # receiving the gaussian with more probability for the pattern
        y = [0]*len(self.gaussians)
        for i in range(len(self.gaussians)):
            y[i] = self.posteriorProbability(x, i)
        gaussian = np.argmax(y)    
                    
        # returning the label
        return gaussian, self.gaussians[gaussian].label
    
def main():
    
    np.random.seed(0)
    
    i = 3
    dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    _, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    stream_records = np.asarray(stream_records[:300], dtype='float')
    x = stream_records[:,0:-1]
    
    gmm = GMM()
    gmm.fitClustering(x, 7)
    #gmm.trainEM(50, True)
    gmm.animation(50)
   
if __name__ == "__main__":
    main()        
            
            
            
            
        