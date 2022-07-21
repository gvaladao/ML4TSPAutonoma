import sys
import math
import time
import pandas as pd
import datetime
import opt_utils
import numpy as np
from collections import namedtuple
sys.setrecursionlimit(10500)    


## This script generates a set of knapsack problems and solves them by using a simple branch-and-bound algorithm (if the computation of the solution takes more than a timeout then the solution obtained is approximate).
## A dataframe with the results is created and saved into the files.

number_of_items = 10
capacity = 1
timeout = 100
max_value = 1
numbasevalues = 10 # The number of values in the base list from which we generate the list of weights and values for the n items of the knapsacks
readKnapsacks = False # Whether knapsacks should be read from file (True) or generated in the script (False)
usegurobi = True # whether we use gurobi (True) or our own BnB algorithm to solve the knapsack
max_weight = capacity # We can consider, without loss of generality, that the capacity is equal to 1 and that the possible maximum value of any weight is the capacity (1). If some variable
                      # is greater than the capacity that means that that variable must be zero and we then have another knapsack with one less variable.

kis = opt_utils.KnapsackInstances(number_of_items, capacity, max_weight,max_value,numbasevalues)

if readKnapsacks:
  dfkis = pd.read_pickle("./dfkis.pkl")
else:
  dfkis = kis.generateKnapsacks()

dfresult = pd.DataFrame(columns=["Value","Certificate","Variables","Elapsed Time"])
number_of_instances = int(dfkis.shape[0]/number_of_items)




for i in range(number_of_instances):
  items = []
  for j in range(0, number_of_items):
    w = dfkis.iloc[i*number_of_items+j][0]
    v = dfkis.iloc[i*number_of_items+j][1]
    d = v/w
    items.append([w,v,d])
  
  # Process each knapsack instance separatly
  items_sorted = sorted(items,key=lambda itm: itm[2],reverse=True)
  aa  = [x for x,y in sorted(enumerate(items), key = lambda itm: itm[1][2],reverse=True)]
  aaa = np.argsort(aa) # This is for afterwards put the variables in their original order (altered when we made the sorted operation)
    
  
  values = np.array([item[1] for item in items_sorted])
  initialOptimisticEstimate = np.sum(values)
  initialState = [0, capacity, initialOptimisticEstimate]
  bestSolution = [-1, []]
  startTime = time.time()
  
  # Check whether the solution to the knapsack problem is trivial
  items_aux = np.array(items)
  if (np.min(items_aux[:,0]) > capacity):
    solution = [0, [0]*number_of_items]
  elif (np.sum(items_aux[:,0])<=capacity):
    solution = [np.sum(items_aux[:,1]), [1]*number_of_items]
  else: # The solution is not a trivial one.
    if usegurobi:
      solution = opt_utils.useGurobi(items_sorted)  
    else:  
      solution = opt_utils.BnB(0, items_sorted, initialState,[],bestSolution,startTime,timeout)
    
  print('i: %s j: %s' %(i ,j))
  value = solution[0]
  endTime = time.time()
  elapsedTime = endTime-startTime

  # Prepare the solution in the specified output format. We check if the solution is certified as optimal and we put the variables in the 
  # original order (their order was changed at the begining when we ordered them by their density). 
  if ((solution[1][-1] == 5) and len(solution[1])>=number_of_items+1): # We must check whether len(solution[1])>=number_of_items+1) because if the timeout is too low the solution might not still have enough elements.
    output_data = str(value) + ' ' + str(0) + '\n' 
    taken = solution[1]
    taken = taken[0:number_of_items]
    taken = np.array(taken)
    taken = taken[aaa] # Put the variables in the original order.
    taken = np.where(taken==-0,0,taken) # Sometimes the optimization solver (e.g. Gurobi) gives the value -0. to a decision variable which is a problem later. 
                                        # Accordingly, we replace those values with 0..
    taken = list(taken)
    row_to_append = {"Value": solution[0],"Certificate": 0, "Variables": taken, "Elapsed Time": elapsedTime} # This solution is not certified as optimal.
  elif (len(solution[1])>=number_of_items): # We must check whether len(solution[1])>=number_of_items) because if the timeout is too low the solution might not still have enough elements.
    output_data = str(value) + ' ' + str(1) + '\n'
    taken = solution[1]
    taken = np.array(taken)
    taken = taken[aaa] # Put the variables in the original order.
    taken = np.where(taken==-0,0,taken) # Sometimes the optimization solver (e.g. Gurobi) gives the value -0. to a decision variable which is a problem later. 
                                        # Accordingly, we replace those values with 0..
    taken = list(taken)
    row_to_append = {"Value": solution[0],"Certificate": 1, "Variables": taken, "Elapsed Time": elapsedTime} # This solution is certified as optimal.
  else:
    taken = 'We could not produce a solution because timeout is too low.'
    row_to_append = {"Value": -500,"Certificate": 0, "Variables": taken, "Elapsed Time": elapsedTime}
  dfresult = dfresult.append(row_to_append,ignore_index=True)

if not readKnapsacks:
  dfkis.to_pickle("nitems_"+str(number_of_items)+"_ninstances_" +str(number_of_instances) + "_" + "dfkis.pkl")

dfresult.to_pickle("nitems_"+str(number_of_items)+"_ninstances_" +str(number_of_instances) + "_" + "dfresult.pkl")