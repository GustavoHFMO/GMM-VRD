#-*- coding: utf-8 -*-

'''
Created on 22 de set de 2018
By Gustavo Oliveira
Universidade Federal de Pernambuco, Recife, Brasil
E-mail: ghfmo@cin.ufpe.br

IMPLEMENTATION OF:
Dynamic Selection Based Drift Handler (Dynse)

P. R. Almeida, L. S. Oliveira, A. S. Britto Jr, and
R. Sabourin, “Adapting dynamic classifier selection for
concept drift,” Expert Systems with Applications, vol.
104, pp. 67–85, 2018.
'''

# Importing dynamic selection techniques:
from data_streams.adjust_labels import Adjust_labels
from competitive_algorithms.prequential_super import PREQUENTIAL_SUPER
al = Adjust_labels()
from streams.readers.arff_reader import ARFFReader
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.dcs.lca import LCA
from deslib.dcs.ola import OLA 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import copy
plt.style.use('seaborn-whitegrid')
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


class PrunningEngine:
    def __init__(self, Type):
        '''
        classe para instanciar o tipo de poda do dynse
        :param: type: tipo da poda [age, accuracy]
        '''
        self.TYPE = Type

    def prunning(self, P, W, C, D):
        '''
        metodo para podar a quantidade de classificadores
        :param: P: pool de classificadores
        :param: W: janela com as instancias a serem avaliadas
        :param: C: novo classificador a ser adicionado
        :param: D: tamanho maximo do pool
        '''
        
        if(self.TYPE=='age'):
            return self.ageBased(P, W, C, D)
        elif(self.TYPE=='accuracy'):
            return self.accuracyBased(P, W, C, D)
    
    def ageBased(self, P, W, C, D):
        '''
        metodo para podar a quantidade de classificadores baseado no classificador mais antigo
        :param: P: pool de classificadores
        :param: W: janela com as instancias a serem avaliadas
        :param: C: novo classificador a ser adicionado
        :param: D: tamanho maximo do pool
        '''
            
        # adicionando um novo classificador ao pool
        P.append(C)
        
        # removendo o classificador mais antigo
        if(len(P)> D):
            del P[0]
                
        return P
    
    def accuracyBased(self, P, W, C, D):
        '''
        metodo para podar a quantidade de classificadores baseado no classificador com menor desempenho
        :param: P: pool de classificadores
        :param: W: janela com as instancias a serem avaliadas
        :param: C: novo classificador a ser adicionado
        :param: D: tamanho maximo do pool
        '''

        # adicionando um novo classificador ao pool
        P.append(C)
        
        # processo para remover o classificador
        if(len(P)> D):
                    
            # ajustando a janela de validacao
            new_W = W[0]
            for i in range(1, len(W)):
                new_W = np.concatenate((new_W, W[i]), axis=0)
            
            # dados para verificar a acuracia dos modelos
            x = new_W[:,0:-1]
            y = new_W[:,-1]
            
            # computando a acuracia de todos os modelos em W
            acuracia = []
            for classificador in P:
                y_pred = classificador.predict(x)
                acuracia.append(accuracy_score(y, y_pred))
                
            # excluindo o classificador com pior desempenho
            del P[np.argmin(acuracia)]
                
        return P

class ClassificationEngine:
    def __init__(self, Type):
        '''
        classe para instanciar o tipo de mecanismo de classificacao do dynse
        :param: type: tipo da poda ['knorae', 'knorau', 'ola', 'lca', 'posteriori', 'priori']
        '''
        self.TYPE = Type
        
    def fit(self, x_sel, y_sel, P, k):
        '''
        metodo para chamar o tipo de DS
        :param: x_sel: dados de treinamento da janela de validacao
        :param: y_sel: rotulos da janela de validacao
        :param: P: pool de classificadores
        :param: k: vizinhanca
        '''
        
        # escolhendo a tecnica de selecao de classificadores
        if(self.TYPE=='knorae'):
            DS = KNORAE(P, k)
        elif(self.TYPE=='knorau'):
            DS = KNORAU(P, k)
        elif(self.TYPE=='ola'):
            DS = OLA(P, k)
        elif(self.TYPE=='lca'):
            DS = LCA(P, k)
        elif(self.TYPE=='posteriori'):
            DS = APosteriori(P, k)
        elif(self.TYPE=='priori'):
            DS = APriori(P, k)
            
        # encontrando os classificadores competentes do DS escolhido
        self.DS = copy.deepcopy(DS)           
        self.DS.fit(x_sel, y_sel)
        
    def predict(self, x):
        '''
        metodo para realizar a predicao com o tipo de classificador selecionado
        :param: x: variaveis de entrada    
        :return: labels referentes a entrada x
        '''
        
        return self.DS.predict(x)
    
class Dynse(PREQUENTIAL_SUPER):
    def __init__(self, D=25, M=100, K=5, train_size=50):
        '''
        Dynamic Selection Based Drift Handler Framework
        :param: D: tamanho maximo do pool
        :param: M: tamanho da janela de estimacao de acuracia
        :param: K: tamanho da vizinhanca
        :param: CE: mecanismo de classificacao
        :param: PE: mecanismo de poda
        :param: BC: classificador base
        '''
        
        self.D = D
        self.M = M
        self.K = K
        self.CE = ClassificationEngine('priori')
        self.PE = PrunningEngine('age')
        self.BC = GaussianNB()
        self.train_size = train_size
        self.NAME = "Dynse-"+self.CE.TYPE+"-"+self.PE.TYPE
        
        # auxiliar variable
        self.PREDICTIONS = []
        self.TARGET = []
        self.count = 0

    def adjustingWindowBatch(self, W):
        '''
        metodo para ajustar a janela de validacao
        :param: W: janela de validacao
        '''
        
        # ajustando a janela de validacao
        new_W = W[0]
        for i in range(1, len(W)):
            new_W = np.concatenate((new_W, W[i]), axis=0)
        
        # dados para treinar
        x = new_W[:,0:-1]
        y = new_W[:,-1]
        
        # retornando os dados
        return x, y
    
    def adjustingWindowOne(self, W):
        '''
        metodo para ajustar a janela de validacao
        :param: W: janela de validacao
        '''
        
        # ajustando a janela de validacao
        new_W = np.asarray(W)

        # dados para treinar
        x = new_W[:,0:-1]
        y = new_W[:,-1]
        
        # retornando os dados
        return x, y
     
    def dividingPatternLabel(self, B):
        '''
        metodo para dividir os dados do batch em treinamento e Exp1
        :param: B: batch a ser dividido
        :param: batch_train: tamanho do batch para treinamento 
        '''
        
        x, y = B[:, 0:-1], B[:,-1]
        
        return x, y
    
    def trainNewClassifier(self, BC, B_train):
        '''
        metodo para treinar um classificador 
        :param: BC: classificador base a ser utilizado
        :param: B: batch a ser treinado
        '''
        
        #obtendo os dados para treinamento e o de Exp1
        x, y = self.dividingPatternLabel(B_train)

        # fazendo uma copia do classe do classificador
        C = copy.deepcopy(BC)
        
        # treinando o classificador
        C.fit(x, y)
        
        # retornando
        return C
        
    def removeOldestBatch(self, W):
        '''
        metodo para remover o batch mais antigo
        :param: W: janela que ira remover o mais antigo
        '''
        
        del W[0]
        
    def prequential_batch(self, labels, stream, step_size, train_size):
        '''
        metodo para executar o codigo
        :param: labels: rotulos existentes no stream
        :param: stream: fluxo de dados
        :param: batch_size: tamanho dos batches
        '''

        # salvando o stream e o tamanho do batch
        self.STREAM = al.adjustStream(labels, stream)
        
        # janela inicial
        W = []
        
        # pool inicial de classificadores
        P = []
        
        # for para percorrer a stream
        for i in range(0, len(self.STREAM), step_size):
            
            # obtendo o atual batch
            B = self.STREAM[i:i+step_size]
            

            # Etapa com dados rotulados ##############################
            
            # obtendo os dados rotulados
            B_train = B[:train_size]
            
            # adicionando o batch na janela
            W.append(B_train)
            
            # treinando um classificador 
            C = self.trainNewClassifier(self.BC, B_train)
                    
            # podando o numero de classificadores
            P = self.PE.prunning(P, W, C, self.D)
                
            # verificando o tamanho da janela
            if(len(W) > self.M):
    
                # removendo o batch mais antigo 
                self.removeOldestBatch(W)

            
            # Etapa com dados nao rotulados ###########################
                    
            # obtendo os dados nao rotulados
            B_test = B[train_size:]
            
            # ajustando a janela de validacao
            x_sel, y_sel = self.adjustingWindowBatch(W)
                    
            # ajustando o mecanismo de classificacao
            self.CE.fit(x_sel, y_sel, P, self.K)
                
            # realizando a classificacao de cada instancia em B
            for x in B_test:
                    
                # recebendo o atual padrao e o seu rotulo
                pattern, label = np.asarray([x[0:-1]]), x[-1]
                    
                # realizando a classificacao
                y_pred = self.CE.predict(pattern)
                
                # salvando a previsao e o alvo
                self.PREDICTIONS.append(y_pred[0])
                self.TARGET.append(label)
                
            # printando a execucao
            self.printIterative(i)
    
    def run(self, labels, stream, cross_validation=False, fold=5, qtd_folds=30):
        '''
        metodo para executar o codigo
        :param: labels: rotulos existentes no stream
        :param: stream: fluxo de dados
        :param: batch_size: tamanho dos batches
        '''

        # salvando o stream e o tamanho do batch
        self.STREAM = al.adjustStream(labels, stream)
        
        # janela inicial
        W = []
        
        # pool inicial de classificadores
        P = []
    
        # variable to store patterns for train
        L = []
        
        # for para percorrer a stream
        for i, X in enumerate(self.STREAM):

            # to use the cross validation
            run=False
            if(cross_validation and self.cross_validation(i, qtd_folds, fold)):
                run = True
            
            # to execute the prequential precedure
            if(run):
                
                # split the current example on pattern and label
                x, y = X[0:-1], int(X[-1])
                                
                # storing the patterns
                L.append(X)
                
                # working to fill the window
                W.append(X)
                    
                # working with full window
                if(i >= self.train_size):
                    
                    # ajustando a janela de validacao
                    x_sel, y_sel = self.adjustingWindowOne(W)
                            
                    try:
                        # ajustando o mecanismo de classificacao
                        self.CE.fit(x_sel, y_sel, P, self.K)
                    except:
                        # to avoid problems with deslib
                        unique = np.unique(y_sel)
                        labels = np.unique(self.STREAM[:,-1])
                        index = [i for i in labels if(i not in unique)] 
                        y_sel[0] = index[0]
                        self.CE.fit(x_sel, y_sel, P, self.K)
                    
                    # realizando a classificacao
                    y_pred = self.CE.predict(np.asarray([x]))
                        
                    # salvando a previsao e o alvo
                    if(i >= self.M):
                        self.PREDICTIONS.append(y_pred[0])
                        self.TARGET.append(y)
                    
                    # training a new classifier
                    if(len(L) > self.train_size):
                        # treinando um classificador 
                        C = self.trainNewClassifier(self.BC, np.asarray(L))
                        # erasing patterns
                        L = []
                        # podando o numero de classificadores
                        P = self.PE.prunning(P, W, C, self.D)
                            
                    # removendo o batch mais antigo 
                    self.removeOldestBatch(W)
                    
                else:
                    # treinando um classificador 
                    C = self.trainNewClassifier(self.BC, np.asarray(L))
                    # erasing patterns
                    L = []
                    # podando o numero de classificadores
                    P = self.PE.prunning(P, W, C, self.D)    
                    
                # printando a execucao
                self.printIterative(i)
            
def main():
    
    
    #1. importando o dataset
    i = 1
    #dataset = ['powersupply', 'PAKDD', 'elec', 'noaa']
    #labels, _, stream_records = ARFFReader.read("../data_streams/real/"+dataset[i]+".arff")
    dataset = ['circles', 'sine1', 'sine2', 'virtual_5changes', 'virtual_9changes', 'SEA', 'SEARec']
    labels, _, stream_records = ARFFReader.read("../data_streams/_synthetic/"+dataset[i]+".arff")
    
    dynse = Dynse(M=50)
    dynse.run(labels, stream_records, cross_validation=True, fold=2, qtd_folds=30)
    
    # printando a acuracia final do sistema
    print(dynse.accuracyGeneral())
    
    # salvando a predicao do sistema
    df = pd.DataFrame(data={'predictions': dynse.PREDICTIONS})
    df.to_csv("../projects/"+dynse.NAME+"-"+dataset[i]+".csv")
        
if __name__ == "__main__":
    main()        

