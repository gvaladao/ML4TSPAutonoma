# ML4TSPAutonoma

This is the private repo of a research effort in the area of ML applied to optimization/operations research.

## Knapsack Problem

The file knapsackGenerationAndSolution.py does 2 things:

1. Generates a set of knapsack problems
2. Solves the generated knapsack problems using a branch and bound basic algorithm. The algorithm is exact, thus it yields a certified optimal solution. However, if the time taken to compute exceeds a certain timeout (user defined) the solution is not certified as being optimal and thus is approximate in general.

To use this code you just have to run the script and have the opt_utils.py file in the same directory as knapsackGenerationAndSolution.py    
