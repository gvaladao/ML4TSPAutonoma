import os
import sys
import pickle
import numpy as np
import common
import matplotlib.pyplot as plt

# Notice that for this script to run there must be a folder named Experiments on the current folder (the folder where this script is located)

np.set_printoptions(threshold=sys.maxsize)

approx_files = sorted([f for f in os.listdir('./Experiments/') if (f.startswith('20')) and ('exact' not in f) and ('query' not in f) ])
exact_files = sorted([f for f in os.listdir('./Experiments/') if (f.startswith('20')) and 'exact' in f])
queryknapsack_files = sorted([f for f in os.listdir('./Experiments/') if (f.startswith('20')) and 'query' in f])
number_of_knapsacksqueries = len(approx_files)

max_ratios = -9*np.ones(number_of_knapsacksqueries) # array to store the maximum ratio (among the nearest neighbor sols) (approx value/exact value) for each query knapsack.
num_exacts = 0
num_higher_than98 = 0

for i in range(number_of_knapsacksqueries):
  
  os.chdir('./Experiments/')
  
  with open(approx_files[i],'rb') as f:
    approx = pickle.load(f) # List of tuples regarding approximate nearest neighbors. Each tuple has 1. knapsack 2. Its exact solution 3. Its exact value.

  with open(exact_files[i],'rb') as f:
    exact = pickle.load(f)

  with open(queryknapsack_files[i],'rb') as f:
    queryknapsack = pickle.load(f) # list of tuples (weights, values) for the query knapsack (the weights and values are normalized).
  
  os.chdir('./..')  
  
  exact_value = exact[0]
  exact_solution = exact[1]
  exact_solution= np.array(list(map(int,exact_solution))) # The solution comes out with floats (from some solvers), thus we convert to int (as the solutions are binary).
  #print(exact_solution)

  exact_profits = np.array([elem[1] for elem in queryknapsack])
  exact_weights = np.array([elem[0] for elem in queryknapsack])

  list_approx_sols = np.array(list(map(lambda x: list(map(int,x[1])),approx))) # Getting an np array whose lines are solutions of approximate nearest neighbors (These Knapsacks all have the same hash, which is the approximately nearest neighbor of the hash of the query knapsack).
  list_approx_values = np.array([elem[2] for elem in approx]) # Get an np array with the values corresponding to the list_approx_solutions 
  list_approx_knapsacks = np.array([elem[0] for elem in approx]) 
  #print(list_approx_sols)
  #print(len(list_approx_sols))

  dists = [common.hamming(exact_solution,app) for app in list_approx_sols] # Distances between exatc solutions of approximate neighbors and the exact solution for the query knapsack

  #print(dists)

  min_distance = np.min(dists)
  indexes_min_distances = np.flatnonzero(dists == min_distance)

  # Compute the values of the query knapsack when we feed it the approximate solutions (list_approx_sols)
  queryknapsack_approximate_values = np.array([np.dot(app,exact_profits) if np.dot(app,exact_weights) <= 1.0 else -9 for app in list_approx_sols]) # If the approximate solution is not feasible to the query knapsack, then que corresponding queryknapsack aproximate value is set to -9 (to signal the infeasibility of the solution/value).
  max_value = np.max(queryknapsack_approximate_values)
  indexes_max_values = np.flatnonzero(queryknapsack_approximate_values == max_value)


  ##### Uncomment to see extra information #####
  # print('QueryKnapsack Exact Value: %f' %exact_value)
  # print('QueryKnapsack Approximate Values: %s' %str(queryknapsack_approximate_values))
  # print('QueryKnapsack Approximate Values Ratios : %s' %str(queryknapsack_approximate_values/exact_value))
  # print('Indexes of the maxima in the QueryKnapsack Approximate Values: %s' %str(indexes_max_values))
  # print('Indexes of the minima in the distances between QueryKnapsack exact solution and Approximate solutions: %s' %str(indexes_max_values))
  ##### Uncomment to see extra information #####
  num_exacts=num_exacts+1 if max_value/exact_value == 1.0 else num_exacts
  num_higher_than98 = num_higher_than98+1 if max_value/exact_value > 0.98 else num_higher_than98  
  max_ratios[i] = max_value/exact_value # If there is a negative ratio that means that for the corresponding query knapsack none of the approximate solutions is feasible.

print("Number times that the approximate algorithm gives the exact maximum: " + str(num_exacts))
print("Number times that the approximate maximum is higher than 98 percent of the exact maximum: " + str(num_higher_than98))

plt.figure(1)
plt.style.use('seaborn-whitegrid')
plt.plot(max_ratios,'o',color='black')
title = 'Performance of the approximation for '+ str(number_of_knapsacksqueries) + ' knapsacks'
plt.title(title)
plt.ylabel('Ratio Approx Max Value/ Exact Maximum Value')
plt.xlabel('Query Knapsacks')
plt.savefig('performance'+str(number_of_knapsacksqueries)+'ks.png')
plt.show()
print('Done!')