import os
import yaml
import common
import pickle
import numpy as np
import more_itertools
import pandas as pd
import opt_utils
from datetime import datetime

# Notice that for this script to run there must be a folder named Experiments on the current folder (the folder where this script is located)


with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
print(config)

number_of_queryknapsacks = config['number_of_queryknapsacks'] # Number of query knaspsacks to generate and for retrieval of the corresponding approximate nearest neighbors.
num_knapsack_pairs = config['num_knapsack_pairs'] # The total number of pairs of knapsacks considered (among the set of pre solved knapsacks). These pairs are randomly sampled from all the possible combinations of 2 knapsacks.
number_of_items = config['number_of_items'] # The number of items of the knapsacks considered. 
number_of_instances = config['number_of_instances']
b = config['b'] # The number of bits used in the hashes.
nper = config['nper'] # The number of permutations used, unless b! is smaller (in this case the number of permutations used is b!).
number_of_permutations = min(nper,np.math.factorial(b)) # The number of permutations of the list of elements to be searched (list of solved and hashed knapsacks). See "Similarity Estimation Techniques from Rounding
                              # Algorithms" by M. Charikar (2002). 
dfresult = pd.read_pickle("nitems_"+str(number_of_items)+"_ninstances_" +str(number_of_instances) + "_" + "dfresult.pkl")
read_permutations = config['read_permutations']

for ks in np.arange(number_of_queryknapsacks):
  if ks >=1:
    read_permutations = True # If we have already generated a query knapsack and made the corresponding retrieval of approximate neighbors then all informations about permutations already exists in the disk and we can read it from there.  
  # knapsack instance randomly generated
  capacity = 30
  max_value = 100
  weights = np.random.randint(1,capacity+1,size=number_of_items)
  values = np.random.randint(max_value+1,size=number_of_items)
  weights_normalized = weights/capacity
  values_normalized = values/max_value
  densities = values_normalized/weights_normalized
  # print("ks: ",ks)
  indices_for_sorting = np.argsort(densities)
  indices_for_sorting = indices_for_sorting[::-1] # To get first the indices of the bigger densities (to sort by descending order)
  weights_normalized = weights_normalized[indices_for_sorting]
  values_normalized = values_normalized[indices_for_sorting]
  query_knapsack = np.concatenate((weights_normalized,values_normalized))



  # Finding the hash that is approximately nearest to the query hash
  if ks < 1: # We need to load the following 4 arrays only once
    with open("nitems_"+str(number_of_items)+ "_ninstances_" +str(number_of_instances) + "_b_" +str(b)+"_Hashes"+".npy", 'rb') as f:
      hashes = np.load(f) # hashes is np array with shape (b,ninstances)

    with open("nitems_"+str(number_of_items)+ "_ninstances_" +str(number_of_instances) + "_b_" +str(b)+"_R"+".npy", 'rb') as f:
      R = np.load(f) # R has shape (d,b) where d = 2 x number_of_items

    with open("num_knapsack_pairs_"+str(num_knapsack_pairs)+ "_ninstances_" +str(number_of_instances) +"_A"+".npy",'rb') as f:
      A = np.load(f)  # A is a d x d matrix, where d = 2 x number_of_items

    with open("nitems_"+str(number_of_items)+ "_ninstances_" +str(number_of_instances)+"_X"+ ".npy",'rb') as f:
      X = np.load(f)

  query_knapsack_hashed = common.h_explicit(query_knapsack,A,R)

  distances_above_and_below = np.array([]) # Contains the distances between the 2 nearest hashed knapsacks for each of the permutations
  indices_of_the_above_and_below = np.array([],dtype=int) # The indices (indices refer to the order in nonpermuted array of knapsacks) of the knapsacks considered in distances_above_and_below. 

  if not read_permutations: # Remember that this can only happen for the first query knapsack, because as long as ks >= 1 we always read permutations from file.
    array_of_sigmas = -9*np.ones((b, number_of_permutations), dtype=int) # A 2D array to store each of the permutations. Each permutation will be a column of the matrix. The intialization to -9 signals that it was not stored yet.
    # set_hashes_sigm_sorted is a 3D array containing hashes sorted. For each layer we have a b x ninstances 2D array with the permuted hashes sorted. The number of layers is the number of permutations.
    set_hashes_sigm_sorted = -9*np.ones((number_of_permutations, b, number_of_instances), dtype=int) # Initialization
    set_hashes_sigm_dec_sorted = -9*np.ones((number_of_permutations, number_of_instances), dtype=int)
    set_hashes_sigm_dec_aux_sorted = -9*np.ones((number_of_permutations, number_of_instances), dtype=int) # A 2D matrix that will store in each line the indexes of the instances of an ascending sort of the corresponding hashes. Each line corresponds to a different permutation.  
  else:
    if ks < 1: # if ks >= 1 then the arrays already exist
      with open('ArrayOfSigmas_' +"NumberOfPermutations_" + str(number_of_permutations) + "_b_" +str(b)+".npy",'rb') as f:
        array_of_sigmas = np.load(f)
      with open('Set_hashes_sigm_sorted_' +"numberofinstances_"+str(number_of_instances)+"_NumberOfPermutations_" + str(number_of_permutations) + "_b_" +str(b)+".npy",'rb') as f:
        set_hashes_sigm_sorted = np.load(f)
      with open('Set_hashes_sigm_dec_sorted_' +"numberofinstances_"+str(number_of_instances)+"_NumberOfPermutations_" + str(number_of_permutations) +".npy",'rb') as f:
        set_hashes_sigm_dec_sorted = np.load(f)
      with open('set_hashes_sigm_dec_aux_sorted_' +"numberofinstances_"+str(number_of_instances)+"_NumberOfPermutations_" + str(number_of_permutations) +".npy",'rb') as f:
        set_hashes_sigm_dec_aux_sorted = np.load(f)


  for i in np.arange(number_of_permutations):
    if not read_permutations:
      sigm = np.array(more_itertools.random_permutation(range(b))) # Generates a tuple of lists that have the indexes of permutations of b elements (the elements are the bits)
      array_of_sigmas[:,i] = sigm
      hashes_sigm = hashes[sigm,:] # A hash matrix with the order of the bits (rows) defined by the permutation sigm  
      query_knapsack_hashed_sigm = query_knapsack_hashed[sigm] # The query knapsack with the elments permuted accordingly to the permutation sigm
      hashes_sigm_dec =  np.array([common.bin2int(x[::-1]) for x in hashes_sigm.T ]) # Gives a list of the decimal numbers corresponding to each of the binary numbers in the hashes 
      hashes_sigm_dec_aux = np.arange(len(hashes_sigm_dec)) # (1 x ninstances array)
      sort_inds = np.argsort(hashes_sigm_dec) # The indexes of the instances of an ascending sort 
      hashes_sigm_dec_sorted = hashes_sigm_dec[sort_inds] # Sorts according to the previous indexes 
      hashes_sigm_dec_aux_sorted = hashes_sigm_dec_aux[sort_inds] # We apply to the indices the permutation of elements produced by the sort. (1 x ninstances array)
      
      query_knapsack_hashed_sigm_dec = common.bin2int(query_knapsack_hashed_sigm[::-1]) # The permuted query knapsack hash  converted to decimal.
      
      
      ind = np.searchsorted(hashes_sigm_dec_sorted,query_knapsack_hashed_sigm_dec,side='left') # ind is the index of the decimal corresponding to the query_knapsack (permuted according to sigm). 
                                                                                              # It is such that ind-1 is the index of immediately lower (relative to the query knapsack) decimal hash. 
      hashes_sigm_reverse_sorted = ((hashes_sigm_dec_sorted.reshape(-1,1) & (2**np.arange(b))) != 0).astype(int) # Converts the list of the decimal hashes into a 2D array containing the hashes (each line is a different hash). The binary hash are in the RMSB (reversed order). Shape of the array: (ninstances x b)
      hashes_sigm_sorted = hashes_sigm_reverse_sorted[:,::-1] # Gets the binary numbers corresponding to each hash (lines of the matrix) in the L-MSB order (usual binary order). Each hash is a line. Shape of the array: (ninstances x b). 
      hashes_sigm_sorted_v2 = hashes_sigm_sorted.T # Gets the binary numbers corresponding to each hash (lines of the matrix) in the L-MSB order (usual binary order). Each hash is a column. Shape of the array: (b x ninstances).
      set_hashes_sigm_sorted[i,:,:] = hashes_sigm_sorted_v2
      set_hashes_sigm_dec_sorted[i,:] = hashes_sigm_dec_sorted
      set_hashes_sigm_dec_aux_sorted[i,:] = hashes_sigm_dec_aux_sorted

    else:  
      sigm = array_of_sigmas[:,i]
      hashes_sigm_sorted = set_hashes_sigm_sorted[i,:,:].T # Gets the binary numbers corresponding to each hash (lines of the matrix) in the usually employed bit order: left-MSB
      hashes_sigm_dec_sorted = set_hashes_sigm_dec_sorted[i,:]
      query_knapsack_hashed_sigm = query_knapsack_hashed[sigm] # The query knapsack with the elments permuted accordingly to the permutation sigm
      
      query_knapsack_hashed_sigm_dec = common.bin2int(query_knapsack_hashed_sigm[::-1]) # The permuted query knapsack hash  converted to decimal.
      
      ind = np.searchsorted(hashes_sigm_dec_sorted,query_knapsack_hashed_sigm_dec,side='left') # ind is the index of the decimal corresponding to the query_knapsack (permuted according to sigm). 
                                                                                               # It is such that ind-1 is the index of immediately lower (relative to the query knapsack) decimal hash. 
      hashes_sigm_dec_aux_sorted = set_hashes_sigm_dec_aux_sorted[i,:]


    if ind!=0:
      below_hash = hashes_sigm_sorted[ind-1]
    else: 
      below_hash = np.array([])

    if ind!=len(hashes_sigm_sorted):
      above_hash = hashes_sigm_sorted[ind]
    else:
      above_hash = np.array([])

    if below_hash.size!=0:
      distances_above_and_below = np.append(distances_above_and_below,common.hamming(below_hash,query_knapsack_hashed))
      indices_of_the_above_and_below = np.append(indices_of_the_above_and_below,hashes_sigm_dec_aux_sorted[ind-1])
    if above_hash.size!=0:
      distances_above_and_below = np.append(distances_above_and_below, common.hamming(above_hash,query_knapsack_hashed))
      indices_of_the_above_and_below = np.append(indices_of_the_above_and_below, hashes_sigm_dec_aux_sorted[ind])



  if not read_permutations:
    with open('ArrayOfSigmas_' +"NumberOfPermutations_" + str(number_of_permutations) + "_b_" +str(b)+".npy",'wb') as f:
      np.save(f,array_of_sigmas)
    with open('Set_hashes_sigm_sorted_' +"numberofinstances_"+str(number_of_instances)+"_NumberOfPermutations_" + str(number_of_permutations) + "_b_" +str(b)+".npy",'wb') as f:
      np.save(f,set_hashes_sigm_sorted)
    with open('Set_hashes_sigm_dec_sorted_' +"numberofinstances_"+str(number_of_instances)+"_NumberOfPermutations_" + str(number_of_permutations) +".npy",'wb') as f:
      np.save(f,set_hashes_sigm_dec_sorted)
    with open('set_hashes_sigm_dec_aux_sorted_' +"numberofinstances_"+str(number_of_instances)+"_NumberOfPermutations_" + str(number_of_permutations) +".npy",'wb') as f:
      np.save(f,set_hashes_sigm_dec_aux_sorted)

  ind_aux = np.argmin(distances_above_and_below)
  ind_nearest = indices_of_the_above_and_below[ind_aux] # Gives the original index of the hash that is the approximately nearest to the query hash. 

  hash_nearest = hashes[:,ind_nearest] # The hash of the approximately nearest knapsack

  # Geting the knapsacks that possess the nearest hash
  temp = hashes.T == hash_nearest
  temp = np.prod(temp,axis=1)
  indexes_of_knapsacks_with_nearest_hash = np.flatnonzero(temp==1)

  list_of_nearest_knapsacks_solutions_and_values = [] # List of tuples regarding approximate nearest neighbors. Each tuple has 1. knapsack 2. Its exact solution 3. Its exact optimal value.
  for i in indexes_of_knapsacks_with_nearest_hash:
    a_nearest_knapsack = X[:,i]  
    a_nearest_knapsack_solution = dfresult.iloc[i][2]
    a_nearest_knapsack_value = dfresult.iloc[i][0]
    list_of_nearest_knapsacks_solutions_and_values.append((a_nearest_knapsack,a_nearest_knapsack_solution,a_nearest_knapsack_value))

  dt = datetime.now().strftime('%Y-%m-%d_%H:%M:%S,%f')

  os.chdir('./Experiments/')
  with open(dt+"_nitems_"+str(number_of_items)+ "_ninstances_" +str(number_of_instances) + "_b_" +str(b)+"_nearest_solutions"+".data",'wb') as f:
    pickle.dump(list_of_nearest_knapsacks_solutions_and_values, f) # List of tuples regarding approximate nearest neighbors. Each tuple has 1. knapsack 2. Its exact solution 3. Its exact optimal value.

  query_knapsack_items = list(zip(weights_normalized,values_normalized))
  query_knapsack_solution = opt_utils.useGurobi(query_knapsack_items)

  with open(dt+ "_nitems_"+str(number_of_items)+ "_queryknapsack"+".data",'wb') as f:
    pickle.dump(query_knapsack_items, f)

  with open(dt+ "_nitems_"+str(number_of_items)+ "_exactsolution"+".data",'wb') as f:
    pickle.dump(query_knapsack_solution, f)

  os.chdir('./..')  