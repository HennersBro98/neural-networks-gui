import numpy as np
from numpy import array as ar
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy.special import expit
from copy import deepcopy

### Neural network object
class NN(object):
    
    def __init__(self,biases,edges,
                 Lambda=0,
                 avgNormalisation=0,featureScaling=1,
                 activationKind='Sigmoid',weightInitType='Xavier'):
        self.biases = biases
        self.edges = edges
        self.Lambda = Lambda
        self.avgNormalisation = avgNormalisation
        self.featureScaling = featureScaling
        self.activationKind = activationKind
        self.noLayers = len(biases)
        self.noHiddenLayers = self.noLayers-2
        self.layers = {}
        for i in range(0,self.noLayers):
            self.layers[i] = layer(self.biases[i],
                                   self.edges[i-1] if i != 0 else None,
                                   edgesNext = (self.edges[i] if i!=self.noLayers-1 else None),
                                   isInputLayer = (i==0),
                                   isOutputLayer = (i==self.noLayers-1))
            self.layers[i].initialiseWeights(weightInitType)
    
    ### Set up weights of whole network    
    def initialiseWeights(self,weightInitType='Xavier'):
        for i in range(1,self.noLayers):
            self.layers[i].initialiseWeights(weightInitType)
        return self.weights
    
    def __setitem__(self,key,value):
        if key >= 0:
            self.layers[key] = value
        else:
            self.layers[self.noLayers+key] = value
        
    def __getitem__(self,key):
        return self.layers[key] if key>=0 else self.layers[self.noLayers+key]
    
    #Following two functions do not allow for setting individual key values, can only change whole thing
    @property
    def weights(self,key=None):
        return {j:self.layers[j].weights for j in range(1,self.noLayers)} if key is None else self.layers[key].weights
    
    @weights.setter
    def weights(self,value):
        for i in range(1,self.noLayers):
            self.layers[i].weights = value[i-1]
    
    def __len__(self):
        return self.noLayers
    
    ### Forward propagation
    def FP(self,X,save_all=False):
        a = [0]*self.noLayers
        a[0] = X.T
        if self[0].bias:
            a[0] = np.insert(a[0],0,1,axis=0)
        for i in range(1,self.noLayers):
            a[i] = activation((self[i].edges*self[i].weights)@a[i-1],self.activationKind)
            if self[i].bias:
                a[i] = np.insert(a[i],0,1,axis=0)
        return [b.T for b in a] if save_all else a[-1].T
    
    ### Backpropagation to copmute gradient of cost function
    def BP(self,X,Y,lambdas=None):
        m = X.shape[0]
        lambdas = self.Lambda if lambdas is None else lambdas
        a = self.FP(X,save_all=True)
        d = [0]*(self.noLayers-1)
        d[-1] = a[-1]-Y
        for l in range(-2,-(self.noLayers-1)-1,-1):                  #think about non-convolutional
            d[l] = (d[l+1]@self[l+1].weights)*a[l]*(1-a[l])        #only for sigmoid
            if self[l].bias:
                d[l] = np.delete(d[l],0,axis=1)
        D = [0]*(self.noLayers-1)
        for l in range(0,self.noLayers-1):
            D[l] = d[l].T@a[l]
        grad_cost = [1/m * (D[i] + lambdas*self[i+1].weights) for i in range(0,self.noLayers-1)]
        for i in range(0,self.noLayers-1):
            if self[i+1].bias:
                grad_cost[i][:,0] = grad_cost[i][:,0] - lambdas/m*self[i+1].weights[:,0]
        return grad_cost
    
    ### Computes cost
    def cost(self,X,Y,lambdas=None):
        lambdas = self.Lambda if lambdas is None else lambdas
        h = self.predict(X)
        if np.any(np.logical_or(np.logical_and(h==0,Y==1),np.logical_and(h==1,Y==0))):
            return 10**9
        checkSame = np.logical_or(np.logical_and(h==0,Y==0),np.logical_and(h==1,Y==1))
        sums = [sum(sum(list(self.weights.values())[i][:,self.biases[i]:]**2)) for i in range(0,len(self.weights.values()))]
        return 1/X.shape[0]*(-sum(sum(Y*np.log(np.where(checkSame,1,h))+(1-Y)*np.log(np.where(checkSame,1,1-h))))+lambdas/2*sum(sums)) #only for sigmoid
    
    ### Models neural network
    def model(self,X,Y,
              avgNormalisation=None,featureScaling=None,alreadyPreprocessed=False,
              lambdas=None,
              trainCVTestSplit=ar([0.6,0.2,0.2]),
              initLoops=1,shuffleLoops=1,weightInitType='Xavier',
              optimiser='Gradient descent',maxIters=10**3,tol=10**-6,
              gradDesc_alpha=1,gradDesc_alphaChange=0,
              steepConj_maxIters=100,steepConj_tol=10**-9,
              gradCheck=False,gradCheckEps=10**-6,costsPlot=False):
        
        if avgNormalisation is not None:
            self.avgNormalisation = avgNormalisation
        if featureScaling is not None:
            self.featureScaling = featureScaling
        if not alreadyPreprocessed:
            X = (X-self.avgNormalisation)/self.featureScaling
        
        lambdas = [self.Lambda] if lambdas is None else lambdas
        if any(l<0 for l in lambdas):
            raise ValueError('lambdas must all be >=0')
            
        trainCVTestSplit = trainCVTestSplit/sum(trainCVTestSplit)
        
        hyperParams = {key:value for key,value in zip(locals(),locals().values()) if (key == 'lambdas')}
        
        ### Shuffles training data
        split1, split2 = int(round(trainCVTestSplit[0]*len(X))), int(round((1-trainCVTestSplit[2])*len(X)))
        if split1 == 0:
            raise ValueError('Not enough training data')
        elif split2 == len(X) and shuffleLoops > 1:
            raise ValueError('Not enough test data')
        elif split1 == split2 and len(lambdas) > 1:
            raise ValueError('Not enough cross validation data')   
        XShuffled, YShuffled = deepcopy(X), deepcopy(Y)

        ### Loops through different shuffles, weight initialisations and regularisation parameters
        test_costs = np.zeros(shuffleLoops)
        bestJ = [0]*shuffleLoops
        IJ_weights = [[0 for x in range(len(lambdas))] for y in range(shuffleLoops)]
        IJ_gradCost,\
            IJ_cost,\
                IJ_gradCheck = deepcopy(IJ_weights),deepcopy(IJ_weights),deepcopy(IJ_weights)
        for i in range(0,shuffleLoops):
            shuffleTogether(XShuffled,YShuffled)
            XShuffledTrain,XShuffledCV,XShuffledTest = np.split(XShuffled,[split1,split2])
            YShuffledTrain,YShuffledCV,YShuffledTest = np.split(YShuffled,[split1,split2])
            
            CV_costs = [0]*len(lambdas)
            for j in range(0,len(lambdas)):
                initLoops_weights,initLoops_finalCost,initLoops_gradCost,initLoops_cost,initLoops_gradCheck = [0]*initLoops,[0]*initLoops,[0]*initLoops,[0]*initLoops,[0]*initLoops
                for k in range(0,initLoops):
                    self.initialiseWeights(weightInitType)
                    
                    initLoops_weights[k],\
                        initLoops_gradCost[k],\
                            initLoops_finalCost[k],\
                                initLoops_cost[k],\
                                    initLoops_gradCheck[k] = self.optimise(XShuffledTrain,YShuffledTrain,kind=optimiser,
                                                                           lambdas=lambdas[j],
                                                                           maxIters=maxIters,tol=tol,
                                                                           gradDesc_alpha=gradDesc_alpha,gradDesc_alphaChange=gradDesc_alphaChange,
                                                                           steepConj_maxIters=steepConj_maxIters,steepConj_tol=steepConj_tol,
                                                                           gradCheck=gradCheck,gradCheckEps=gradCheckEps,costsPlot=costsPlot)

                bestK = np.argmin(initLoops_finalCost)
                IJ_weights[i][j] = initLoops_weights[bestK]
                IJ_gradCost[i][j] = initLoops_gradCost[bestK]
                IJ_cost[i][j] = initLoops_cost[bestK]
                IJ_gradCheck[i][j] = initLoops_gradCheck[bestK]
                self.weights = list(IJ_weights[i][j].values())
                CV_costs[j] = self.cost(XShuffledCV,YShuffledCV,lambdas[j]) if len(lambdas)>1 else 0          

            bestJ[i] = np.argmin(CV_costs)
            self.weights = list(IJ_weights[i][bestJ[i]].values())
            test_costs[i] = self.cost(XShuffledTest,YShuffledTest,lambdas[bestJ[i]]) if shuffleLoops>1 else 0
        
        bestI = np.argmin(test_costs)
        self.weights = list(IJ_weights[bestI][bestJ[bestI]].values())
        hyperParams['lambdas'], self.Lambda = lambdas[bestJ[bestI]], lambdas[bestJ[bestI]]
        accuracy = self.accuracy(XShuffledTest,YShuffledTest) if split2 < len(X) else None
        
        if costsPlot:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            costsGraph = plt.figure()
            axes = costsGraph.add_axes([0.1,0.1,0.9,0.9])
            for c in IJ_cost[bestI]:
                axes.plot(c)
            axes.set_title(r'Costs via Optimiser on Training Set using Best Shuffle')
            axes.set_xlabel(r'Optimiser Iteration')
            axes.set_ylabel(r'Cost')
            axes.legend(lambdas)  
            
        return hyperParams,bestJ[bestI],(IJ_gradCost[bestI][bestJ[bestI]],IJ_gradCheck[bestI][bestJ[bestI]]),(IJ_cost[bestI],costsGraph if costsPlot else None),accuracy
    
    ### Gradient descent on weights
    def gradientDescent(self,X,Y,lambdas=None,
                        tol=10**-6,maxIters=10**3,
                        alpha=0.1,alphaChange=0,
                        gradCheck=False,gradCheckEps=10**-6,costsPlot=False):    #Think about contraints if non-convolutional
        lambdas = self.Lambda if lambdas is None else lambdas
        iters = 0
        alphaIn = alpha
        if costsPlot:
            costs = [self.cost(X,Y,lambdas)]
        gradCost = self.BP(X,Y,lambdas)
        while np.sqrt(sum([(g**2).sum() for g in gradCost])) >= tol and iters < maxIters:
            self.weights = [w-alpha*g for w,g in zip(self.weights.values(),gradCost)]   
            gradCost = self.BP(X,Y,lambdas)
            iters += 1
            alpha = alphaIn/(1+alphaChange*iters)
            if costsPlot:
                costs.append(self.cost(X,Y,lambdas)) 
        if costsPlot:
            finalCost = costs[-1]
        else:
            finalCost = self.cost(X,Y,lambdas)
        if gradCheck:
            gradCheckVal = self.gradientCheck(X,Y,lambdas,gradCheckEps)
        return self.weights,gradCost,finalCost,(costs if costsPlot else None),(gradCheckVal if gradCheck else None)
    
    ### Function to find optimal step size in steepest descents or conjugate gradients
    def optimiserPhi(self,X,Y,lambdas,D,l):
        origWeights = deepcopy(self.weights)
        self.weights = [w+l*d for w,d in zip(self.weights.values(),D)]
        ret = self.cost(X,Y,lambdas)
        self.weights = [x for x in list(origWeights.values())]
        return ret
    
    ### Steepest descents or conjugate gradients on weights
    def steepConj(self,X,Y,lambdas=None,optimiser='Conjugate gradients',
                  tol=10**-6,maxIters=10**2,
                  steepConj_maxIters=10,steepConj_tol=10**-9,
                  gradCheck=False,gradCheckEps=10**-6,costsPlot=False):
        lambdas = self.Lambda if lambdas is None else lambdas
        iters = 0
        if costsPlot:
            costs = [self.cost(X,Y,lambdas)]
        f = lambda l: self.optimiserPhi(X,Y,lambdas,D,l) 
        G = self.BP(X,Y,lambdas)
        while np.sqrt(sum([(g**2).sum() for g in G])) >= tol and iters < maxIters:
            if iters % 2 == 0 or optimiser == 'Steepest descents':
                D = [-g for g in G]
            else:
                GPrev = [(gprev**2).sum() for gprev in GPrev]
                beta = sum([(g**2).sum()/np.where(gprev==0,1,gprev) for g,gprev in zip(G,GPrev)])
                D = [-g+beta*d for g,d in zip(G,D)]   
            lBest = minimize(f,0,tol=steepConj_tol,options={'maxiter':steepConj_maxIters}).x
            self.weights = [w+lBest*d for w,d in zip(self.weights.values(),D)]
            if optimiser == 'Conjugate gradients':
                GPrev = deepcopy(G)
            G = self.BP(X,Y,lambdas)
            iters += 1
            if costsPlot:
                costs.append(self.cost(X,Y,lambdas))
        if costsPlot:
            finalCost = costs[-1]
        else:
            finalCost = self.cost(X,Y,lambdas)
        if gradCheck:
            gradCheckVal = self.gradientCheck(X,Y,lambdas,gradCheckEps)
        return self.weights,G,finalCost,(costs if costsPlot else None),(gradCheckVal if gradCheck else None)
    
    ### General optimiser with different options
    def optimise(self,X,Y,lambdas=None,kind='Conjugate gradients',
                 tol=10**-6,maxIters=10**3,
                 gradDesc_alpha=0.1,gradDesc_alphaChange=0,
                 steepConj_maxIters=10,steepConj_tol=10**-9,
                 gradCheck=False,gradCheckEps=10**-6,costsPlot=False):
        if kind == 'Gradient descent':
            return self.gradientDescent(X,Y,lambdas=lambdas,
                                        tol=tol,maxIters=maxIters,
                                        alpha=gradDesc_alpha,alphaChange=gradDesc_alphaChange,
                                        gradCheck=gradCheck,gradCheckEps=gradCheckEps,costsPlot=costsPlot)
        elif kind == 'Steepest descents' or kind == 'Conjugate gradients':
            return self.steepConj(X,Y,lambdas=lambdas,optimiser=kind,
                                  tol=tol,maxIters=maxIters,
                                  steepConj_maxIters=steepConj_maxIters,steepConj_tol=steepConj_tol,
                                  gradCheck=gradCheck,gradCheckEps=gradCheckEps,costsPlot=costsPlot)
    
    ### Gradient check function
    def gradientCheck(self,X,Y,lambdas,gradCheckEps):
        gradCheckVal = deepcopy(self.weights)
        for i in range(1,self.noLayers):
            for j in range(0,self[i].weights.shape[0]):
                for k in range(0,self[i].weights.shape[1]):
                    orig = np.copy(self[i].weights[j,k])
                    self[i].weights[j,k] += gradCheckEps
                    upper = self.cost(X,Y,lambdas)
                    self[i].weights[j,k] -= 2*gradCheckEps
                    lower = self.cost(X,Y,lambdas)
                    self[i].weights[j,k] = orig
                    gradCheckVal[i][j,k] = (upper-lower)/(2*gradCheckEps)
        return list(gradCheckVal.values())
    
    ### Predict with showing confidences    
    def predict(self,X,alreadyPreprocessed=True):
        if not alreadyPreprocessed:
            X = (X-self.avgNormalisation)/self.featureScaling
        return self.FP(X,save_all=False)
    
    ### Predict whether in or out of each class
    def predictRounded(self,X,alreadyPreprocessed=True):
        return np.round(self.predict(X,alreadyPreprocessed))
    
    ### If each record can only be in a single class: predicts class
    def predictClass(self,X,alreadyPreprocessed=True):
        return ar([self.predict(X,alreadyPreprocessed).argmax(axis=1)+1]).T
    
    ### Accuracy of model
    def accuracy(self,X,Y,alreadyPreprocessed=True):
        return (1-np.abs((Y-self.predictRounded(X,alreadyPreprocessed))).max(axis=1)).sum()/X.shape[0]

### Layer object
class layer(object):
    def __init__(self,bias,edges,isInputLayer=False,isOutputLayer=False,edgesNext=None):
        self.bias = bias
        self.edges = edges
        self.edgesNext = edgesNext
        self.isInputLayer = isInputLayer
        self.isOutputLayer = isOutputLayer
        self.no_nodes = edgesNext.shape[1] if isInputLayer else edges.shape[0] + bias    #May need to be changed if non-convolutional
        self.no_inputs = self.no_nodes-bias if self.isInputLayer else edges.sum()
        self.no_outputs = (0 if isOutputLayer else sum(edgesNext).sum())
    
    ### Initialise weights of the layer           
    def initialiseWeights(self,weightInitType='Xavier'):
        if not self.isInputLayer:
            if weightInitType == 'Classic':
                self.weights = 1/np.sqrt(self.no_inputs)*np.random.randn(self.edges.shape[0],self.edges.shape[1])
            elif weightInitType == 'Xavier':
                self.weights = 2*np.sqrt(6)/np.sqrt(self.no_inputs+self.no_outputs)*(np.random.rand(self.edges.shape[0],self.edges.shape[1])-0.5)
            elif weightInitType == 'Kaiming':
                self.weights = np.sqrt(2/self.no_inputs)*np.random.randn(self.edges.shape[0],self.edges.shape[1])
            else:
                raise ValueError()
    
    def __setitem__(self,key,value):
        x,y = key
        self.weights[x][y] = value
    
    def __getitem__(self,key):
        x,y = key
        return self.weights[x][y]
    
    def __len__(self):
        return self.no_nodes

### Computes activation function
def activation(z,kind='Sigmoid',zero_is=0.5):
    if kind == 'Sigmoid':
        return expit(z)
    elif kind == 'Heaviside':
        return np.heaviside(z,zero_is)
    elif kind == 'ReLU':
        return np.maximum(z,0)
    else:
        raise ValueError('\'', kind, '\' is an invalid activation function')

### Extracts data and provides info for feature scaling and average normalisation
def extract_data(X,Y=None,Y_on=True,featureScaling=None,avgNormalisation=None): 
    m, n = X.shape[0], X.shape[1]
    if Y_on:
        K = Y.shape[1]
    else:
        K = None
    
    if avgNormalisation == 'Mean':
        avgs = np.mean(X,axis=0)
    elif avgNormalisation == 'Median':
        avgs = np.median(X,axis=0)
    elif avgNormalisation == 'Mode':
        avgs = np.mode(X,axis=0)
    else:
        avgs = np.zeros(X.shape[1])
    
    if featureScaling == 'Range':
        scales = X.ptp(axis=0)
    elif featureScaling == 'Standard Deviation':
        scales = X.std(axis=0)
    elif featureScaling == 'Variance':
        scales = X.var(axis=0)
    else:
        scales = np.ones(X.shape[1])
        
    scales = np.where(scales!=0,scales,1)  
    return X, Y, m, n, K, avgs, scales

### Shuuffles datasets
def shuffleTogether(x,y):
    randState = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(randState)
    np.random.shuffle(y)

if __name__ == '__main__':
    pass