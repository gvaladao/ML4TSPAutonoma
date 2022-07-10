import random
import itertools
import numpy as np
import time
import datetime
import pandas as pd


## A small library of utilities to discrete optimization tasks

def BnB(depth, valuesWeightsDensities, state, vars, bestSolution, startTime, timeout):
    ## Simple branch and bound algorithm to solve the knapsack problem. If the computations finish below a timeout time the solution is certifiable optimum. Otherwise the solution is approximate.

    print('Depth: %s\n' % depth)
    # print('Vars: %s\n' %vars)

    weight, value, dummy1 = valuesWeightsDensities[depth]

    newValue1 = state[0] + value  # State Value (x_i = 1)
    newValue0 = state[0]  # State Value (x_i = 0)
    newRoom1 = state[1] - weight  # State Room
    newRoom0 = state[1]  # State Room
    newVars1 = vars + [1]
    newVars0 = vars + [0]
    auxVars1 = np.array([newVars1]) == 0
    auxVars0 = np.array([newVars0]) == 0
    valuesDepth = np.array([item[1] for item in valuesWeightsDensities[
                                                :depth + 1]])  # The values of items until index equals depth (in the list of items ordered, decreasingly, by density).
    values = np.array([item[1] for item in valuesWeightsDensities])
    newOptimisticEstimate1 = np.sum(values) - np.dot(auxVars1,
                                                     valuesDepth)  # Optimistic Estimate    # To obtain the optimistic estimate we must discount the values of the items that will not enter the knapsack,
    # from the sum of all values.
    newOptimisticEstimate1 = newOptimisticEstimate1[0]  # We want a value and not an array
    newOptimisticEstimate0 = np.sum(values) - np.dot(auxVars0, valuesDepth)  # Optimistic Estimate
    newOptimisticEstimate0 = newOptimisticEstimate0[0]
    atFullDepth = depth > (len(valuesWeightsDensities) - 2)
    newState1 = [newValue1, newRoom1, newOptimisticEstimate1]
    newState0 = [newValue0, newRoom0, newOptimisticEstimate0]
    value, room, optimisticEstimate = state

    endTime = time.time()
    if (endTime - startTime > timeout):
        bestSolution[1] = bestSolution[1] + [5]
        return bestSolution

    if (not atFullDepth):
        if (newRoom1 >= 0):
            if newOptimisticEstimate1 >= bestSolution[0]:
                # print('newVars1: %s\n' %newVars1)
                solution1 = BnB(depth + 1, valuesWeightsDensities, newState1, newVars1, bestSolution, startTime,
                                timeout)
                aux = bool(solution1)
                print('Depth1: %s\n' % depth)
                if aux and (solution1[0] > bestSolution[0]):
                    bestSolution[0] = solution1[0]
                    bestSolution[1] = solution1[1]
        if (newRoom0 >= 0):
            if newOptimisticEstimate0 >= bestSolution[0]:
                # print('newVars0: %s\n' %newVars0)
                solution2 = BnB(depth + 1, valuesWeightsDensities, state, newVars0, bestSolution, startTime, timeout)
                aux = bool(solution2)
                print('Depth2: %s\n' % depth)
                if aux and (solution2[0] > bestSolution[0]):
                    bestSolution[0] = solution2[0]
                    bestSolution[1] = solution2[1]
        if ((newRoom1 < 0) or (newOptimisticEstimate1 < bestSolution[0])) and (
                (newRoom0 < 0) or (newOptimisticEstimate0 < bestSolution[0])):
            return False
        else:
            return bestSolution

    if (atFullDepth):
        if (newRoom1 >= 0):
            solution = (newState1[0], newVars1)
            # print('solution1: %s\n' %solution[0])
            # print('NewVars1: %s\n' %newVars1)
            if (solution[0] > bestSolution[0]):
                bestSolution[0] = solution[0]
                bestSolution[1] = solution[1]
        else:
            solution = (newState0[0], newVars0)
            # print('newState0: %s' %newState0[0])
            # print('solution0: %s\n' %solution[0])
            # print('NewVars0: %s\n' %newVars0)
            if (solution[0] > bestSolution[0]):
                bestSolution[0] = solution[0]
                bestSolution[1] = solution[1]

        return bestSolution


class KnapsackInstances:
    # Generates instances of the Knapsack problem
    def __init__(self, n_items, k, max_weight, weight_steps, max_value):
        self.n_items = n_items
        self.k = k
        self.max_weight = max_weight
        self.weight_steps = weight_steps
        self.max_value = max_value
        self.df = pd.DataFrame()

    def generateKnapsacks(self):
        base_weights_list = [i for i in np.arange(1, self.max_weight + 1, self.weight_steps)]
        items_weights = list(itertools.product(base_weights_list, repeat=self.n_items))

        base_values_list = [random.randint(0, self.max_value) for i in base_weights_list]
        items_values = list(itertools.product(base_values_list, repeat=self.n_items))

        # Linearization of lists
        items_weights = [x for tern in items_weights for x in tern]
        items_values = [x for tern in items_values for x in tern]

        # Dataframe creation
        self.df['Weights'] = items_weights
        self.df['Values'] = items_values
        # return items_weights
        return self.df
