#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

import time
import datetime
from collections import namedtuple
import sys

import opt_utils

sys.setrecursionlimit(10500)
Item  = namedtuple("Item", ['index', 'value', 'weight'])
Item2 = namedtuple("Item", [ 'value', 'weight', 'density'])

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
        items2.append(Item2( float(parts[0]), float(parts[1]), float(parts[0])/float(parts[1])))
        items2sorted = sorted(items2,key=lambda itm: itm.weight,reverse=False)
    
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

    weights,values = np.array([item[0:2] for item in items2sorted]).T.astype(int)
    initialOptimisticEstimate = np.sum(values)
    initialState = [0, capacity, initialOptimisticEstimate]
    bestSolution = [-1, []]

    startTime = time.time()
    timeout=100
    st = datetime.datetime.now()
    solution = opt_utils.BnB(0, items2sorted, initialState,[],bestSolution,startTime,timeout)
    et = datetime.datetime.now()
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
    et = datetime.datetime.now()

    elapsedTime = et - st
    print('Elapsed time (s): %s\n' % str(elapsedTime.total_seconds()))
    print(output_data)
    st = datetime.datetime.now()
    max_value, solution=opt_utils.knap_sack_dynamic_programming(capacity,weights,values,item_count)
    print("Recursive function: Max value: ",max_value," solution: ",solution)
    et = datetime.datetime.now()

    elapsedTime = et - st
    print('Elapsed time (s): %s\n' % str(elapsedTime.total_seconds()))

    #return output_data
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
    
    # with open('./moredata/large_scale/knapPI_2_10000_1000_1', 'r') as input_data_file:
    # knapPI_3_10000_1000_1
    #./moredata/low-dimensional/f1_l-d_kp_10_269
    with open('./moredata/low-dimensional/multiple_solutions2', 'r') as input_data_file:


        input_data = input_data_file.read()
        print(solve_it(input_data))

    
