import copy
import random
import pandas as pd
import numpy as np
from ast import BitXor
from numpy import linalg as LA
from itertools import combinations
from more_itertools import random_combination

### Learning to hash knapsack problems.

### Heavily inspired on: Fast Image Search for Learned Metrics. Jain, Kulis, Grauman, CVPR 2008.



#### Learning the Mahalanobis Metric


##### Learning a Mahalanobis Metrics from an information point of view

##### Based on: Information-Theoretic Metric Learning. Davis, Kulis, Jain, Sra, Dhillion, ICML 2007.

####### Input Parameters #######
dfkis = pd.read_pickle("./nitems_10_ninstances_92378_dfkis.pkl")
dfr   = pd.read_pickle("./nitems_10_ninstances_92378_dfresult.pkl")
number_of_items = 10
b = 6 # The number of bits output by the hashing function
lper = 5
uper = 95
tSim = 1 # $ (i,j) \in S \iff dHamming(x_i,x_j) <= tSim $  
tDsim = number_of_items-1 # $ (i,j) \in D \iff dHamming(x_i,x_j) >= tDsim $
gamma = 0.1 # The slack variable parameter (in the ITML paper they refer experimenting with gamma = 0.01, 0.1, 1, 10)
tol = 0.01
niterations = 100000 # The max number of iterations. Achieved if tol threshold is not met before. 
readX = False
num_knapsack_pairs = 1000# The total number of pairs of knapsacks considered. These pairs are randomly sampled from all the possible combinations of 2 knapsacks.
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
comb = [random_combination(np.arange(number_of_instances),2) for aux in range(num_knapsack_pairs)] # A list of the pairs of knapsacks considered. These
# pairs are randomly selected from all the possible pairs of knapsacks. This list has num_knapsack_pairs elements
# We are considering that if the hamming distance between the solutions (vector of decision variables) of 2 knapsacks is equal or lower than 1 then the
# the two knapsacks are similar. When the hamming distance is higher or equal than number_of_items -1 the[n the two knapsacks are dissimilar. 
S = [ pair for pair in comb if np.sum(np.bitwise_xor(np.array(dfr.iloc[pair[0]].tolist()[2]), np.array(dfr.iloc[pair[1]].tolist()[2]))) <= 1 ]
D = [ pair for pair in comb if np.sum(np.bitwise_xor(np.array(dfr.iloc[pair[0]].tolist()[2]), np.array(dfr.iloc[pair[1]].tolist()[2]))) >= number_of_items-1 ]

######### We determine the 5th and 95th percentiles of the distances between at least 100 pairs of knapsacks.

totalNumPairs = len(comb)
numPairs = np.min([100, totalNumPairs])
distances = np.zeros(numPairs)

for t in np.arange(numPairs):
  pairNo = random.randint(0,totalNumPairs-1)
  i = comb[pairNo][0]
  j = comb[pairNo][1]

  v1 = X[:,i]
  v2 = X[:,j]
  distances[t] = v1 @ A0 @ v2 
  
  
######### Compute the histogram
[v,e]=np.histogram(distances,100)
l = e[int(np.floor(lper))]
u = e[int(np.floor(uper))] 


pairsInSandD = S + D
lambdas     = {pair: 0 for pair in pairsInSandD } # lambdas initialization
lambdasOld  = {pair: 0 for pair in pairsInSandD } # lambdas initialization
slacksS = {pair: u for pair in S} # Slack variables initialization
slacksD = {pair: l for pair in D}
slacks = {}
slacks.update(slacksS)
slacks.update(slacksD)


####### Iterations

random.shuffle(pairsInSandD) # We are shuffling the pairs to pick up pairs randomly from S and D along the iterative process.
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


#### Building and applying the hash function

def hExplicit(x,A,b):
  # x: a numpy 1-d array which represents a knapsack. It possesses the weights and values of the knapsack.
  # A: a learned mahalanobis metric in the space of the knapsacks (weights and values).
  # b: the number of bits produced by the hash function
  d = x.size
  GT = np.linalg.cholesky(A)
  G = GT.T
  h = -1*np.ones(b)
  
  mu, sigma = 0, 1.0
  for i in np.arange(b):
    r = np.random.default_rng().normal(mu,sigma,d)
    h[i] = 1 if r.T @ G @ x >= 0 else -1
  
  return h
  
d, ninstances = X.shape

hashes = np.zeros(b,ninstances)

for i in np.arange(ninstances):
  hashes[:,i] = hExplicit(X[:,i],A,b)

    
with open("nitems_"+str(number_of_items)+ "_ninstances_" +str(ninstances) + "_b_" +str(b)+"_Hashes"+".npy",'wb') as f:
  np.save(f,hashes)
    
    
    
  

