import numpy as np



def h_explicit(x,A,R):
  #### A hash function. See: Fast Image Search for Learned Metrics. Jain, Kulis, Grauman, CVPR 2008.  
  # x: a numpy 1-d array which represents a knapsack. It possesses the weights and values of the knapsack.
  # A: a learned mahalanobis metric in the space of the knapsacks (weights and values).
  # R: The matrix for which each column is a realization of a d-dimensional standard normal random variable (mu=0, sigma=1). 
  
  GT = np.linalg.cholesky(A)
  G = GT.T
  dummy, b = R.shape
  h = np.zeros(b)
  for i in np.arange(b):
    r = R[:,i]
    h[i] = 1 if r.T @ G @ x >= 0 else 0
  
  return h

def bin2int(x):
    # x must be a binary number written in the order R-MSB (the opposite of what is usual)
    x=x.astype(int)
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y


def hamming(v,w):
# Computes the Hamming distance between v and w. v and w must be numpy arrays with binary values
  v = v.astype(int)
  w = w.astype(int)
  aux = v!=w
  hamming_dist = aux.sum()
  return hamming_dist