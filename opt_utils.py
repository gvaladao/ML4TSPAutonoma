import random
import itertools
import numpy as np
import time
import datetime
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from more_itertools import random_combination_with_replacement

## A small library of utilities to discrete optimization tasks

def BnB(depth,valuesWeightsDensities,state,vars,bestSolution,startTime,timeout):

## Simple branch and bound algorithm to solve the knapsack problem. If the computations finish below a timeout time the solution is certifiable optimum. Otherwise the solution is approximate.

    print('Depth: %s\n' %depth)
    #print('Vars: %s\n' %vars)
    
    weight, value, dummy1 = valuesWeightsDensities[depth]

    newValue1 = state[0] + value  # State Value (x_i = 1)
    newValue0 = state[0]          # State Value (x_i = 0)
    newRoom1 = state[1] - weight  # State Room
    newRoom0 = state[1]           # State Room
    newVars1 = vars + [1]
    newVars0 = vars + [0]
    auxVars1 = np.array([newVars1])==0
    auxVars0 = np.array([newVars0])==0
    valuesDepth = np.array([item[1] for item in valuesWeightsDensities[:depth+1]]) # The values of items until index equals depth (in the list of items ordered, decreasingly, by density).
    values = np.array([item[1] for item in valuesWeightsDensities])
    newOptimisticEstimate1 = np.sum(values) -  np.dot(auxVars1, valuesDepth) # Optimistic Estimate    # To obtain the optimistic estimate we must discount the values of the items that will not enter the knapsack, 
                                                                                                          # from the sum of all values.
    newOptimisticEstimate1 = newOptimisticEstimate1[0] # We want a value and not an array
    newOptimisticEstimate0 = np.sum(values) -  np.dot(auxVars0, valuesDepth) # Optimistic Estimate
    newOptimisticEstimate0 = newOptimisticEstimate0[0]    
    atFullDepth = depth>(len(valuesWeightsDensities)-2)
    newState1 = [newValue1, newRoom1, newOptimisticEstimate1]
    newState0 = [newValue0, newRoom0, newOptimisticEstimate0]
    value, room, optimisticEstimate = state

        
        
    endTime = time.time()
    if (endTime - startTime > timeout):
        bestSolution[1] = bestSolution[1] + [5]
        return bestSolution


    if (not atFullDepth):
        if (newRoom1  >= 0):
            if newOptimisticEstimate1 >= bestSolution[0]:
                #print('newVars1: %s\n' %newVars1)
                solution1 = BnB(depth+1,valuesWeightsDensities,newState1,newVars1, bestSolution, startTime, timeout)
                aux = bool(solution1)
                print('Depth1: %s\n' %depth)
                if aux and (solution1[0] > bestSolution[0]):
                    bestSolution[0] = solution1[0]
                    bestSolution[1] = solution1[1]
        if (newRoom0 >= 0):
            if newOptimisticEstimate0 >= bestSolution[0]:
                #print('newVars0: %s\n' %newVars0)
                solution2 = BnB(depth+1,valuesWeightsDensities,state, newVars0, bestSolution, startTime, timeout)
                aux = bool(solution2)
                print('Depth2: %s\n' %depth)
                if aux and (solution2[0] > bestSolution[0]):
                    bestSolution[0] = solution2[0]
                    bestSolution[1] = solution2[1]
        if ((newRoom1 < 0) or (newOptimisticEstimate1 < bestSolution[0]) ) and ((newRoom0 < 0) or (newOptimisticEstimate0 < bestSolution[0]) ):
            return False
        else:
            return bestSolution
            
    if (atFullDepth):
        if (newRoom1 >= 0):
            solution = (newState1[0],newVars1)
            #print('solution1: %s\n' %solution[0])
            #print('NewVars1: %s\n' %newVars1)
            if (solution[0] > bestSolution[0]):
                bestSolution[0] = solution[0]
                bestSolution[1] = solution[1]
        else:
            solution = (newState0[0],newVars0)
            #print('newState0: %s' %newState0[0])
            #print('solution0: %s\n' %solution[0])     
            #print('NewVars0: %s\n' %newVars0)
            if (solution[0] > bestSolution[0]):
                bestSolution[0] = solution[0]
                bestSolution[1] = solution[1]

        return bestSolution

def useGurobi(items):
### items is a list of elements [w, v, d] (weights, values, and density)
	n = len(items)
	weights = [element[0] for element in items]
	values = [element[1] for element in items]
	k = 1 # We assume that all the knapsacks are normalized: the weights and the capacity all are divided by the capacity (so the capacity turns 1). As it can be considered that, without loss of generality, no weight is higher than the capacity, then all the weights have a value between 0 and 1.  
	#  The values are all divided by the max value (so all of the values are also in the interval [0  1]).

	m = gp.Model()
	x = m.addVars(n, vtype=GRB.BINARY, name='x')
	m.setObjective(x.prod(values),GRB.MAXIMIZE)
	m.addConstr(x.prod(weights) <= k, name='knapsack')
	m.optimize()
	vars = [v.x for v in m.getVars()]
	objective = m.ObjVal 
	bestSolution = (objective,vars)
	return bestSolution

class KnapsackInstances:
  # Generates instances of the Knapsack problem
	def __init__(self,n_items,k,max_weight,max_value,numbasevalues,num_of_instances):
		self.n_items = n_items
		self.k = k
		self.max_weight = max_weight
		self.max_value = max_value
		self.df = pd.DataFrame()
		self.numbasevalues = numbasevalues # The number of values in the base list from which we generate the list of weights and values for the n items of the knapsacks        
		self.num_of_instances = num_of_instances # The number of knapsack instances to be generated. If num_of_instances is equal to -1 then the number of instances shall be the one given by all the combinations
																						 # of values of base_weights_list with replacement (combinations of numbasevalues to n_items).   
	
	def generateKnapsacks(self):
		base_weights_list = list(np.linspace(1e-2,self.max_weight,num=self.numbasevalues))
		base_values_list = [np.random.random_sample() for i in base_weights_list] # random_sample() gives a value from the “continuous uniform” distribution over [0.0 1.0)
		
		if self.num_of_instances == -1:
			items_weights = np.array(itertools.combinations_with_replacement(base_weights_list,self.n_items)) # List of lists of the weights for each knapsack instance. List of lists of the weights for each knapsack instance. combinations_with_replacement allows us to put aside all the replica lists that only differ in the ordering
			items_values = np.array(itertools.combinations_with_replacement(base_values_list,self.n_items)) # List of lists of the values for each knapsack instance. combinations_with_replacement allows us to put aside all the replica lists that only differ in the ordering
			self.num_of_instances = len(items_weights)
		else:
			items_weights = np.array([random_combination_with_replacement(base_weights_list,self.n_items) for i in np.arange(self.num_of_instances)]) # List of lists of the weights for each knapsack instance. combinations_with_replacement allows us to put aside all the replica lists that only differ in the ordering
			items_values = np.array([random_combination_with_replacement(base_values_list,self.n_items) for i in np.arange(self.num_of_instances)]) # List of lists of the values for each knapsack instance. combinations_with_replacement allows us to put aside all the replica lists that only differ in the ordering
				
		# Sorting the items of the knapsacks by decreasing density
		for i in np.arange(len(items_weights)): # Iterating across the knapsacks lists of weigths
			aux = np.array([])
			for j in np.arange(len(items_weights[i])): # For each knapsack list of weights iterate through each weight
				aux = np.append(aux, items_values[i][j]/items_weights[i][j])
			aux_inds = aux.argsort()[::-1]
			items_weights[i]=items_weights[i][aux_inds]
			items_values[i] = items_values[i][aux_inds]
				
		# Linearization of lists
		items_weights = [x for tern in items_weights for x in tern]
		items_values = [x for tern in items_values for x in tern]

    # Dataframe creation
		self.df['Weights'] = items_weights
		self.df['Values'] = items_values
    #return items_weights
		return self.df