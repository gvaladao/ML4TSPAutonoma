# ML4TSPAutonoma

This is the repo of a research effort in the area of ML applied to optimization/operations research.

The file solver.py contains a branch and bound approach to the knapsack problem. To run you can open a terminal and just run 

    python solver.py

The data that defines the knapsack problem to be solved must be in a text file with data in the following pattern:

    n   k
    v1  w1
    v2  w2
    .   .
    .   .
    .   .
    vn  wn

In the above, the first line contains the number of items (n) and the capacity of the knapsack (k), while the following lines contain the value and weight for each of the n items.

There are a set of low and large scale knapsack problems in files contained in the folders inside the *moredata* folder. To solve a specific problem instance, the path to the corresponding definition file must be set in the code line where the file is read. For example:

     with open('./moredata/large_scale/knapPI_2_10000_1000_1', 'r') as input_data_file:
     
