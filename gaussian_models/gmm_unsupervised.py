'''
Created on 29 de abr de 2018
@author: gusta
'''

from gaussian_models.prototype_selection import PrototypeSelection
ps = PrototypeSelection()
from streams.readers.arff_reader import ARFFReader
from sklearn.neighbors import NearestNeighbors
import matplotlib.patches as patches
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")

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
        plt.title('Time: ' +str(t))
        plt.legend(loc='upper right')
        
        # ajustando o layout
        plt.gcf().set_size_inches(4.5, 4)
        plt.subplots_adjust(bottom=0.08, top=0.99, left=0.08, right=0.99)
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
        
        # adjusting the pattern
        x = np.transpose(np.array([x]))
        mu = np.transpose(np.array([self.mu]))
        
        ################ part 1 #####################
        A = np.power(2*np.pi, len(x)/2)
        B = np.linalg.det(self.sigma)
        
        # dealing with matrix with zero determinant
        if(B == 0): 
            #B = B + 0.0001
            pseudo = True 
        else:
            pseudo = False
        
        # part C
        C = np.sqrt(np.abs(B))
        
        # joining the equation
        part1 = 1/(A * C)
        #############################################
        
        # part 2
        part2 = np.transpose(np.subtract(x, mu))
        
        # part 3
        # dealing with zero on matrix
        if(pseudo == False):
            part3 = np.linalg.inv(self.sigma)
        else:
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

        # pertinence
        return pertinence
    
    def printstats(self):
        '''
        method to print the current mu and sigma of the distribution
        '''
        print("mi:", self.mu)
        print("sigma:", self.sigma)
        
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
        :param: minimum_instances: the minimum number of instances to randomly be chosen 
        '''
        
        # storing the dataset
        self.train_input = train_input
            
        # number of gaussians
        self.K = K
    
        # creating the gaussians
        self.gaussians = []
            
        # allocating itens for each gaussian
        for _ in range(K):
            # selecting randomly one observation to start the EM process
            if(len(self.train_input) > 1):
                index = np.random.randint(0, len(self.train_input)-1)
            else:
                index = 0
            observation = self.train_input[index]
            
            # adding some noisy to avoid the gaussian shrink to this data
            # mean
            tm = len(observation)
            mu = observation + np.random.uniform(0, 1, tm)
            # cov
            cov = (0.05**2) * np.identity(tm)
            cov = cov + np.random.uniform(-1, 1, (tm, tm))
            
            # creating the gaussian
            g = Gaussian(mu, cov, 1./K)
                
            # storing the gaussians
            self.gaussians.append(g)
                
        # defining the quantity of parameters of model
        self.n = len(self.train_input)
        
        # quantity of means, covariances and constants of mixtures
        self.p = self.K * 3
            
        # intializing the matrix of weights
        self.matrixWeights = self.Estep()
        
    def trainEM(self, iterations, criterion=True):
        '''
        method to train the gaussians
        :param: iterations: integer - quantity of iterations necessary to train the models
        :param: log: boolean - variable to show the log of train
        '''
        
        # just update gaussians with many instances
        if(self.n > 1):
        
            # process to train the gmm
            self.listLoglike = []
            for i in range(iterations):
                
                #printing the training
                #print("iteration[", i, "]:")
                
                # before to update the parameters storing the old gaussians
                self.storingOldGaussians()
                
                # EM process
                self.Mstep(self.matrixWeights)
                
                # analysing if the gaussians are collapsing and dealing with it
                self.analysingCollapse()
                
                # continuing the optimization
                self.matrixWeights = self.Estep()
                
                # calculating the new loglikelihood
                self.listLoglike.append(np.abs(self.loglike))
                
                # stop criterion
                if(criterion and (i > 2)):
                    calculo = 100-(self.listLoglike[i]*100/self.listLoglike[i-1])
                    if(calculo < 0.15):
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

            # sum of probabilities, in other words, the sum of instance x belongs to all clusters, on the final is equal to one
            den = np.sum(weights)
            if(den == 0): den = 0.001
            
            # completing the theorem of bayes for all clusters 
            weights /= den
            
            # add into loglike
            self.loglike += np.log(den)
            
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
            if(dens == 0): dens = 0.001
            
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
    
    def storingOldGaussians(self):
        '''
        Method to guarantee that each iteration the last gaussian will be stored
        '''
        self.old_gaussians = copy.deepcopy(self.gaussians)
    
    def analysingCollapse(self):
        '''
        Method to analyse if of one gaussian is collapsing
        '''
        
        # for by iterate over each gaussian
        for i in range(len(self.gaussians)):

            # verifying if the component has all zeros
            calc = np.sum(self.gaussians[i].mu)
            
            # condition to reinitialize the gaussian collapsed
            if(calc == 0):
                tm = len(self.old_gaussians[i].mu)
                self.gaussians[i].mu = self.old_gaussians[i].mu + np.random.uniform(0, 1, tm)
                self.gaussians[i].sigma = self.old_gaussians[i].sigma * np.random.uniform(0, 1, (tm, tm))
                
    def chooseBestModel(self, train_input, type_selection, Kmax, restarts, iterations, stop_criterion=True):
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
        gmm.trainEM(iterations, stop_criterion)
        
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
        total = np.sum(dens) 
        if(total == 0.0): total = 0.01  
        
        # posterior probability
        posterior = self.conditionalProbability(x, i)/total 
        
        # returning the posterior probability
        return posterior

    def posteriorProbabilities(self, x):
        '''
        method to return the posterior probabilities of an variable x to a gaussian i
        :param: x: observation
        '''
    
        # calculating the density
        dens = []
        for j in range(len(self.gaussians)):
            dens.append(self.conditionalProbability(x, j))
        
        # to avoid nan
        dens = np.nan_to_num(dens) 
        total = np.sum(dens) 
        if(total == 0.0): total = 0.01  
        
        # calculating the probability for each gaussian
        probabilities = []
        for i in dens:
            probabilities.append(i/total)
        
        # returning the posterior probability
        return probabilities
        
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
        probabilities = self.posteriorProbabilities(x)
        gaussian = np.argmax(probabilities)    

        # returning the label
        return self.gaussians[gaussian].label
    
    def predict_gaussian(self, x):
        '''
        method to predict the class for a only pattern x
        :param: x: pattern
        :return: the respective label for x
        '''
        
        # receiving the gaussian with more probability for the pattern
        probabilities = self.posteriorProbabilities(x)
        gaussian = np.argmax(probabilities)    
                    
        # returning the label
        return gaussian, self.gaussians[gaussian].label

class GMM_KDN(GMM):
    def __init__(self, noise_threshold=0.7, n_vizinhos=5, kmax=2, emit=20, stop_criterion=True):
        '''
        Constructor of GMM_VD model
        :kdn_train: to activate the use of kdn on training
        :criacao: to activate the creation of gaussians throught the stream
        :tipo_atualizacao: type of update used
        :noise_threshold: the value to define an noise
        :kmax: max number of gaussian used per class
        :n_vizinhos: number of neighboors used on kdn
        '''
        
        self.noise_threshold = noise_threshold
        self.n_vizinhos = 1+n_vizinhos
        self.Kmax = kmax
        self.emit = emit
        self.stop_criterion = stop_criterion
        
    '''
    KDN PRE-PROCESSING
    '''
        
    def easyInstances(self, x, y, limiar, n_vizinhos):
        '''
        Method to return a subset of validation only with the easy instacias
        :param: x: patterns
        :param: y: labels
        :return: x_new, y_new: 
        '''
        
        # to guarantee
        classes_inicial, qtds_inicial = np.unique(np.asarray(y), return_counts=True)
        
        # classes to be maintained
        classes_maintained = [] 
        for i, c in enumerate(classes_inicial):
            if(qtds_inicial[i] < self.n_vizinhos):
                classes_maintained.append(c)
        
        # computing the difficulties for each instance
        dificuldades = self.kDN(x, y)
        
        # variables to save the new instances
        x_new = []
        y_new = []
        # loop by kdn to select instances
        for kdn, p, c in zip(dificuldades, x, y): 
            if(kdn < limiar or c in classes_maintained):
                x_new.append(p)
                y_new.append(c)
                
        # classes to be maintained 2
        classes_final, _ = np.unique(np.asarray(y_new), return_counts=True)
        classes_not_included = [i for i in classes_inicial if i not in classes_final]
        
        # adding the elements excluded
        for p, c in zip(x, y):
            if(c in classes_not_included):
                x_new.append(p)
                y_new.append(c)
                
        # returning the selected data
        return np.asarray(x_new), np.asarray(y_new), classes_inicial, qtds_inicial
    
    def kDN(self, X, Y):
        '''
        Method to compute the hardess of an observation based on a training set
        :param: X: patterns
        :param: Y: labels
        :return: dificuldades: vector with hardness for each instance 
        '''
        
        # to store the hardness
        hardness = [0] * len(Y)

        # for to compute the hardness for each instance
        for i, (x, y) in enumerate(zip(X,Y)):
            hardness[i] = self.kDNIndividual(x, y, X, Y)
        
        # returning the hardness
        return hardness
    
    def kDNIndividual(self, x_query, y_query, x_sel, y_sel, plot=False):
        '''
        Metodo para computar o grau de dificuldade de uma observacao baseado em um conjunto de validacao
        :param: x_query: padrao a ser consultado
        :param: x: padroes dos dados
        :param: y: respectivos rotulos
        :return: dificuldade: flutuante com a probabilidade da instancia consultada 
        '''
    
        # defining the neighboors
        nbrs = NearestNeighbors(n_neighbors=self.n_vizinhos, algorithm='ball_tree').fit(x_sel)
        
        # consulting the next neighboors
        _, indices = nbrs.kneighbors([x_query])
        # removing the query instance
        indices = indices[0][1:]
        
        # verifying the labels
        cont = 0
        for j in indices:
            if(y_sel[j] != y_query):
                cont += 1
                    
        # computing the hardness
        hardness = cont/(self.n_vizinhos-1)
        
        #====================== to plot the neighboors ===================================
        if(plot):
            self.plotInstanceNeighboors(x_query, y_query, hardness, indices, x_sel)
        #==================================================================================
            
        # returning the hardness
        return hardness
    
    '''
    SUPERVISED LEARN
    '''
    
    def fit(self, train_input, train_target, noise_threshold=False, n_vizinhos=False):
        '''
        method to fit the data to be clusterized
        :param: train_input: matrix, the atributes must be in vertical, and the examples in the horizontal - must be a dataset that will be clusterized
        :param: train_target: vector, with the labels for each pattern input
        :param: Kmax: number max of gaussians to test. Default 4
        :param: restarts: integer - number of restarts. Default 1
        :param: iterations: integer - number of iterations to trains the gmm model. Default 30
        '''
        
        # initialization
        if(noise_threshold==False):
            noise_threshold = self.noise_threshold
        elif(n_vizinhos==False):
            n_vizinhos = self.n_vizinhos
        
        # getting only the easy instances
        self.train_input, self.train_target, unique, _ = self.easyInstances(train_input, train_target, noise_threshold, n_vizinhos)
        
        # updating the variables to use for plot
        self.L = len(unique)
        self.unique = list(unique)
        
        # instantiating the gaussians the will be used
        self.gaussians = []
         
        # creating the optimal gaussians for each class
        for y_true in unique:
            
            # dividing the patterns by class
            x_train, _ = self.separatingDataByClass(y_true, self.train_input, self.train_target)
            # training a gmm
            self.trainGaussians(x_train, y_true, kmax=self.Kmax)
        
        # updating the gaussian weights          
        self.updateWeight()
                            
    def separatingDataByClass(self, y_true, x_train, y_train):
        '''
        method to separate data by class
        :y_true: label to be separeted
        :x_train: patterns
        :y_train: labels
        :return: x_train, y_train corresponding y_true
        '''
        
        # getting by class
        X_new, Y_new = [], []
        for x, y in zip(x_train, y_train):
            if(y == y_true):
                X_new.append(x)
                Y_new.append(y)
                
        # returning the new examples
        return np.asarray(X_new), np.asarray(Y_new)
    
    def trainGaussians(self, data, label, type_selection="AIC", kmax=2):
        '''
        method to train just one class
        :label: respective class that will be trained
        :data: data corresponding label
        :type_selection: AIC or BIC criterion
        '''
        
        # EM with AIC applied for each class
        gmm = self.chooseBestModel(data, type_selection, kmax, 1, self.emit, self.stop_criterion)

        # adding to the final GMM the just trained gaussians
        self.addGMM(gmm, label)
        
        # returning the gmm
        return gmm

    def updateWeight(self):
        '''
        Method to update the mix
        '''
        
        # computing the density
        sum_dens = 0 
        for g in self.gaussians:
            sum_dens += g.dens
        if(sum_dens == 0.0): sum_dens = 0.01
        
        # for each gaussian computing its weight
        for i in range(len(self.gaussians)):
            self.gaussians[i].mix = self.gaussians[i].dens/sum_dens

    '''
    SUPPORT METHODS
    '''
    
    def addGMM(self, gmm, y_true):
        '''
        Method to add a new gmm in the final GMM
        :y: respective label of GMM
        :gmm: gmm trained
        '''

        # storing the gaussians            
        for gaussian in gmm.gaussians:
            gaussian.label = y_true 
            self.gaussians.append(gaussian)
            
        # defining the number of gaussians for the problem
        self.K = len(self.gaussians)
    
    def addGaussian(self, gaussian):
        '''
        Method to insert a new gaussian into GMM
        '''
        
        #adding the new gaussian
        self.gaussians.append(gaussian)
        
        # defining the number of gaussians for the problem
        self.K = len(self.gaussians)
 
def main():
    
    # importing the datasets
    i = 4
    dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    stream_records = stream_records[:300]
    
    # adjusting the labels
    from data_streams.adjust_labels import Adjust_labels
    al = Adjust_labels()
    stream_records = al.adjustStream(labels, stream_records)
    
    # training set
    x = stream_records[:,0:-1]
    
    # adjusting the gmm
    gmm = GMM()
    gmm.fitClustering(x, 4)
    #gmm.trainEM(50, True)
    gmm.animation(100)
   
if __name__ == "__main__":
    main()        
            
            
            
            
        