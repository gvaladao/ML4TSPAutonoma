# https://gist.github.com/jhelgert/eb8cc47dda8795f60a68afb10ac95433
from random import randint

from gurobipy import *


def cancel_last_solution(m,j):
    B, NB = [], []

    for i in range(n):
        B.append(i) if (x[i].x == 1) else NB.append(i)
    # It sufices either one of the 1s to go to 0 or one 0 to go to 1 and the next sum will be less than len(B)
    # the constraint forbids the sum to reach len(B)
    m.addConstr((quicksum(x[i] for i in B) - quicksum(x[j] for j in NB)) <= len(B) - 1, name="CutSolution{}".format(j))
    m.update()


# define data coefficients
def solve_problem(p, w, c):  # prices weights, capacity
    # p = [6, 6, 8, 9, 6, 7, 3, 2, 6]
    # w = [2, 3, 6, 7, 5, 9, 4, 8, 5]
    # c = 20
    # w=[1]*30
    # p=w
    # c=2
    n = len(w)
    # Maximize
    #   6 x[0] + 6 x[1] + 8 x[2] + 9 x[3] + 6 x[4] + 7 x[5] + 3 x[6] + 2 x[7] + 6 x[8]
    # Subject To
    #  2 x[0] + 3 x[1] + 6 x[2] + 7 x[3] + 5 x[4] + 9 x[5] + 4 x[6] + 8 x[7] + 5 x[8] <= 20

    solution_dict = {}
    # create empty model
    m = Model()

    # add decision variables
    x: None = m.addVars(n, vtype=GRB.BINARY, name='x')

    # set objective function
    m.setObjective(quicksum(p[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

    # add constraint
    m.addConstr((quicksum(w[i] * x[i] for i in range(n)) <= c), name="Capacity")

    # Add Lazy Constraints
    m.update()

    # quiet gurobi
    m.setParam(GRB.Param.LogToConsole, 0)
    k = 0  # index of the current solution
    m.optimize()  # solve the model
    solution_new = solution_old = m.ObjVal

    while (m.Status == GRB.OPTIMAL) and (solution_old == solution_new):
        k += 1
        # Found new feasible optimal solution
        # m.write("{}.sol".format(k))
        solution_dict[k] = m.getAttr('x')
        solution_old = solution_new
        # Make the previous solution infeasible
        # cancel_last_solution(k)
        B, NB = [], []

        for i in range(n):
            B.append(i) if (x[i].x == 1) else NB.append(i)
        m.addConstr((quicksum(x[i] for i in B) - quicksum(x[j] for j in NB)) <= len(B) - 1, name="CutSolution{}".format(k))
        m.update()
        m.optimize()  # solve the model
        solution_new = m.ObjVal

    print("Found {} optimal feasible Solutions!".format(k))
    # print("Max val: ", solution_old)
    overall_solution = [0] * len(solution_dict[1])
    for i in range(k):
        # print(i + 1, solution_dict[i + 1])
        for j in range(len(solution_dict[i + 1])):
            overall_solution[j] += solution_dict[i + 1][j]
    for j in range(len(solution_dict[1])):
        overall_solution[j] = overall_solution[j] / len(solution_dict)
    # print("overal solution: ",overall_solution)
    # m.write("knapsack.lp")


days_of_the_year = 365
problem_size = 30
generated_problems = 100
for problem in range(generated_problems):
    w = [randint(1, days_of_the_year) for _ in range(problem_size)]
    p = [randint(1,2) for _ in range(problem_size)]
    min_capacity = min(p)
    max_capacity = sum(p)
    c = randint(min_capacity, max_capacity)
    print("problem: ",problem)
    print("w: ", w)
    print("c: ", c)

    solve_problem(p, w, c)
