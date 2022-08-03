import copy
import yaml
import random
import pandas as pd
import numpy as np
from ast import BitXor
from numpy import linalg as LA
from itertools import combinations
import more_itertools 
import common

### Learning to hash knapsack problems.

### Heavily inspired on: Fast Image Search for Learned Metrics. Jain, Kulis, Grauman, CVPR 2008.



#### Learning the Mahalanobis Metric


##### Learning a Mahalanobis Metrics from an information point of view

##### Based on: Information-Theoretic Metric Learning. Davis, Kulis, Jain, Sra, Dhillion, ICML 2007.



with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)


####### Input Parameters #######
dfkis = pd.read_pickle("./nitems_30_ninstances_90000_dfkis.pkl")
dfr   = pd.read_pickle("./nitems_30_ninstances_90000_dfresult.pkl")
number_of_items = config['number_of_items']
b = config['b'] # The number of bits output by the hashing function
lper = config['lper'] # The lower percentile of the distances between a set of pairs to be used in the algorithm that learns the Mahalanobis metric.
uper = config['uper'] # The higher percentile of the distances between a set of pairs to be used in the algorithm that learns the Mahalanobis metric.
tSim = config['tSim'] # $ (i,j) \in S \iff dHamming(x_i,x_j) <= tSim $ The threshold of similarity between 2 knapsacks  
gamma = config['gamma'] # The slack variable parameter (in the ITML paper they refer experimenting with gamma = 0.01, 0.1, 1, 10)
tol = config['tol'] # The tolerance considered in the iterative algorithm that learns the Mahalanobis matrix A. When the difference between 2 certain quantities is less than tol we consider that the algorithm converged. Used in the file ('hashingKnapsacks.py').
niterations = config['niterations'] # The max number of iterations in the iterative algorithm that learns the Mahalanobis matrix A. Achieved if tol threshold is not met before. Used in the file ('hashingKnapsacks.py').
readX = config['readX'] # True if we want to read X from a file. False otherwise.
num_knapsack_pairs = config['num_knapsack_pairs'] # The total number of pairs of knapsacks considered (among the set of pre solved knapsacks). These pairs are randomly sampled from all the possible combinations of 2 knapsacks.
####### Input Parameters #######

####### Definitions #######
number_of_instances = int(dfkis.shape[0]/number_of_items)
####### Definitions #######

####### Initializations
if not readX:
  X = np.zeros((2*number_of_items,number_of_instances)) # d x n matrix where d = 2*number_of_items and n = number_of_instances
  for k,g in dfkis.groupby(np.arange(len(dfkis))//number_of_items): # Write the array X with n columns; each column is the vector of features of each knapsack; the features are composed by the weights 
    # and the values of the items of each knapsack.
    g=g[['Weights','Values']].melt(value_name='inputFeatures')
    g = g[['inputFeatures']]
    h=g.to_numpy()
    X[:,k] =  h.flatten() # We must flatten as h has shape(2*number_of_items,1) and X[:,k] is expecting an array with shape (2*number_of_items)
    print("k: %i" %k)
  with open("nitems_"+str(number_of_items)+ "_ninstances_" +str(number_of_instances)+"_X"+ ".npy",'wb') as f:
    np.save(f,X)
else:
  with open("nitems_"+str(number_of_items)+ "_ninstances_" +str(number_of_instances)+"_X"+ ".npy",'rb') as f:
    X= np.load(f)

A0 = np.eye(2*number_of_items)
A = A0

######### Creation of the sets S and D (the sets of similar and dissimilar items, respectively).
######### Set S contains the pairs of indexes of similar knapsacks and the set D contains the pairs of indexes of dissimilar knapsacks
comb = [more_itertools.random_combination(np.arange(number_of_instances),2) for aux in range(num_knapsack_pairs)] # A list of the pairs of knapsacks considered. These
# pairs are randomly selected from all the possible pairs of knapsacks. This list has num_knapsack_pairs elements
# We are considering that if the hamming distance between the solutions (vector of decision variables) of 2 knapsacks is equal or lower than tSim then the
# the two knapsacks are similar. When the hamming distance is higher than tSim then the two knapsacks are dissimilar. 
S = [ pair for pair in comb if np.sum(np.bitwise_xor(np.array(dfr.iloc[pair[0]].tolist()[2],dtype=int), np.array(dfr.iloc[pair[1]].tolist()[2],dtype=int))) <= tSim]
D = [ pair for pair in comb if np.sum(np.bitwise_xor(np.array(dfr.iloc[pair[0]].tolist()[2],dtype=int), np.array(dfr.iloc[pair[1]].tolist()[2],dtype=int))) >  tSim ]

######### We determine the 5th and 95th percentiles of the distances between at least 100 pairs of knapsacks.

totalNumPairs = len(comb) # Total number of knapsack pairs
numPairs = np.min([len(S), len(D)]) # The number of pairs used to compute the histogram and from there the quantiles
distances = -1*np.ones(numPairs)

for t in np.arange(numPairs):
  pairNo = random.randint(0,totalNumPairs-1)
  i = comb[pairNo][0]
  j = comb[pairNo][1]

  v1 = X[:,i]
  v2 = X[:,j]
  
  #distances[t] = v1 @ A0 @ v2 
  
  vd = v1-v2
  distances[t] = vd @ A0 @ vd 
  
######### Compute the histogram and get the lper and uper quantiles
[v,e]=np.histogram(distances,100)
l = e[int(np.floor(lper))]
u = e[int(np.floor(uper))] 

random.shuffle(S)
random.shuffle(D)
pairsInSandD = list(more_itertools.interleave_longest(S,D)) # This way we interleave elements from D and S and thus we mitigate the possible imbalance between the two classes 
lambdas     = {pair: 0 for pair in pairsInSandD } # lambdas initialization
lambdasOld  = {pair: 0 for pair in pairsInSandD } # lambdas initialization
slacksS = {pair: u for pair in S} # Slack variables initialization
slacksD = {pair: l for pair in D}
slacks = {}
slacks.update(slacksS)
slacks.update(slacksD)


####### Iterations
converged = False
iters = 0
aux=-1

while not converged:
  for pair in pairsInSandD:
    v = X[:,pair[0]]-X[:,pair[1]]
    p = v.T @ A @ v
    delta = 1 if pair in S else -1
    alpha = np.min([lambdas[pair],delta/2*(1/p-gamma/slacks[pair])])
    beta  = delta*alpha/(1-delta*alpha*p)
    slacks[pair] = gamma*slacks[pair]/(gamma + delta*alpha*slacks[pair])
    lambdas[pair] = lambdas[pair] - alpha
    v = np.expand_dims(v,axis=1)
    A = A + beta*A @ v @ v.T @ A
    iters+=1
    if iters%5000==0 and aux != -1:
      print("Converg: %f, Iterations: %d" % (aux, iters))  
  lambdasVec     = np.array(list(lambdas.values()))
  lambdasOldVec  = np.array(list(lambdasOld.values()))
  normlam = LA.norm(lambdasVec)
  if (normlam==0):
    break
  else:
    aux = LA.norm(lambdasVec-lambdasOldVec)/normlam 
    lambdasOld = copy.deepcopy(lambdas)
    if (aux < tol) or (iters > niterations):
      converged = True

with open("num_knapsack_pairs_"+str(num_knapsack_pairs)+ "_ninstances_" +str(number_of_instances) +"_A"+".npy",'wb') as f:
  np.save(f,A)

  
d, ninstances = X.shape

hashes = np.zeros((b,ninstances))

R = np.zeros((d,b)) # 2D matrix having b columns and where each column is a d-dimensional vector that is a realization of a (d-dimensional) normal distribution (mu = 0 and sigma = 1).
mu, sigma = 0, 1.0
for j in np.arange(b):
  R[:,j] = np.random.default_rng().normal(mu,sigma,d)

with open("nitems_"+str(number_of_items)+ "_ninstances_" +str(ninstances) + "_b_" +str(b)+"_R"+".npy",'wb') as f:
  np.save(f,R)


for i in np.arange(ninstances):
  hashes[:,i] = common.h_explicit(X[:,i],A,R)

    
with open("nitems_"+str(number_of_items)+ "_ninstances_" +str(ninstances) + "_b_" +str(b)+"_Hashes"+".npy",'wb') as f:
  np.save(f,hashes)
    
    

  

