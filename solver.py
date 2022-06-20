#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time
import datetime
from collections import namedtuple
import sys
sys.setrecursionlimit(10500)    
Item  = namedtuple("Item", ['index', 'value', 'weight'])
Item2 = namedtuple("Item", ['index', 'value', 'weight', 'density'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items  = []
    items2 = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, float(parts[0]), float(parts[1])))
        items2.append(Item2(i-1, float(parts[0]), float(parts[1]), float(parts[0])/float(parts[1])))
        items2sorted = sorted(items2,key=lambda itm: itm.density,reverse=True)
    
    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    # value = 0
    # weight = 0
    # taken = [0]*len(items2sorted)

    # for item in items2sorted:
    #     if weight + item.weight <= capacity:
    #         taken[item.index] = 1
    #         value += item.value
    #         weight += item.weight
    
    
    
    
    ######### Branch & Bound with the relaxation of the capacity constraint #########

    def BnB(depth,valuesWeightsDensities,state,vars,bestSolution,startTime):

        print('Depth: %s\n' %depth)
        print('Vars: %s\n' %vars)
        
        dummy1, value, weight, dummy2 = valuesWeightsDensities[depth]

        newValue1 = state[0] + value  # State Value (x_i = 1)
        newValue0 = state[0]          # State Value (x_i = 0)
        newRoom1 = state[1] - weight  # State Room
        newRoom0 = state[1]           # State Room
        newVars1 = vars + [1]
        newVars0 = vars + [0]
        auxVars1 = np.array([newVars1])==0
        auxVars0 = np.array([newVars0])==0
        valuesDepth = np.array([item[1] for item in valuesWeightsDensities[:depth+1]])
        values = np.array([item[1] for item in valuesWeightsDensities])
        newOptimisticEstimate1 = np.sum(values) -  np.dot(auxVars1, valuesDepth) # Optimistic Estimate    
        newOptimisticEstimate1 = newOptimisticEstimate1[0]
        newOptimisticEstimate0 = np.sum(values) -  np.dot(auxVars0, valuesDepth) # Optimistic Estimate
        newOptimisticEstimate0 = newOptimisticEstimate0[0]    
        atFullDepth = depth>(len(valuesWeightsDensities)-2)
        newState1 = [newValue1, newRoom1, newOptimisticEstimate1]
        newState0 = [newValue0, newRoom0, newOptimisticEstimate0]
        value, room, optimisticEstimate = state

        
        
        endTime = time.time()
        if (endTime - startTime > 100):
                bestSolution[1] = bestSolution[1] + [5]
                return bestSolution


        if (not atFullDepth):
            if (newRoom1  >= 0):
                if newOptimisticEstimate1 >= bestSolution[0]:
                    #print('newVars1: %s\n' %newVars1)
                    solution1 = BnB(depth+1,valuesWeightsDensities,newState1,newVars1, bestSolution, startTime)
                    aux = bool(solution1)
                    
                    if aux and (solution1[0] > bestSolution[0]):
                        bestSolution[0] = solution1[0]
                        bestSolution[1] = solution1[1]
            if (newRoom0 >= 0):
                if newOptimisticEstimate0 >= bestSolution[0]:
                    #print('newVars0: %s\n' %newVars0)
                    solution2 = BnB(depth+1,valuesWeightsDensities,state, newVars0, bestSolution, startTime)
                    aux = bool(solution2)
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
                print('solution1: %s\n' %solution[0])
                print('NewVars1: %s\n' %newVars1)
                if (solution[0] > bestSolution[0]):
                    bestSolution[0] = solution[0]
                    bestSolution[1] = solution[1]
            else:
                solution = (newState0[0],newVars0)
                #print('newState0: %s' %newState0[0])
                print('solution0: %s\n' %solution[0])     
                print('NewVars0: %s\n' %newVars0)
                if (solution[0] > bestSolution[0]):
                    bestSolution[0] = solution[0]
                    bestSolution[1] = solution[1]

            return bestSolution
    
    values = np.array([item[1] for item in items2sorted])
    initialOptimisticEstimate = np.sum(values)
    initialState = [0, capacity, initialOptimisticEstimate]
    bestSolution = [-1, []]
    startTime = time.time()
    solution = BnB(0, items2sorted, initialState,[],bestSolution,startTime)
    value = solution[0]
    
    # prepare the solution in the specified output format
    if (solution[1][-1] == 5):
        output_data = str(value) + ' ' + str(0) + '\n'
        taken = solution[1]
        taken = taken[0:item_count]
    else:
        output_data = str(value) + ' ' + str(1) + '\n'
        taken = solution[1]
    output_data += ' '.join(map(str, taken))
    return output_data    
    ######### Branch & Bound with the relaxation of the capacity constraint #########


if __name__ == '__main__':
    # import sys
    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     with open(file_location, 'r') as input_data_file:
    #         input_data = input_data_file.read()
    #     print(solve_it(input_data))
    # else:
    #     print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
    
    with open('./moredata/large_scale/knapPI_2_10000_1000_1', 'r') as input_data_file:

        st = datetime.datetime.now()                        
        input_data = input_data_file.read()
        print(solve_it(input_data))
        et = datetime.datetime.now()                        

        elapsedTime = et-st
        print('Elapsed time (s): %s\n' %str(elapsedTime.total_seconds()))
    
