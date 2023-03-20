import os

# f=os.listdir("../knapsackProblemInstances/problemInstances/n_400_c_1000000_g_2_f_0.1_eps_0.1_s_100/")
from datetime import datetime

import opt_utils
name="n_400_c_1000000_g_2_f_0.1_eps_0.1_s_100"
file = open( "../knapsackProblemInstances/problemInstances/"+name+"/test.in", 'r')
'''
format first items, then (id, p w) then capacity
3
1 3 8
2 2 8
3 9 1
10
"

This describes a problem instance in which there are 
n=3 items and the knapsack has a capacity of 
c=10. The item with 
id 1 has a profit of 3 and a weight of 8.
The item with id 2 has a profit of 2 and a weight of 8. The item with id 3 has a profit of 9 and a weight of 1.

400 
0 600092 600056
1 600035 600079
...
398 77 38
399 14 44
1000000

'''
line_no = 0;
n=0
c=0
problem = {"items": []}  # a dictionary with various propertie + items a list of dicts.
problem["name"]=name
while True:
    line = file.readline()
    if not line:
        break
    line = line.replace("\n", "")
    if line_no==0:
        problem["n"]=int(line)
    elif line_no==problem["n"]+1:
        problem["c"]=int(line)
    else:
        content = line.split(" ")
        item = {"id_x": int(content[0]), "p": int(content[1]), "w": int(content[2])}
        problem["items"].append(item)
    line_no += 1
file.close()
problem["sorted_items"] = sorted(problem["items"], key=lambda x: (-x['p'] / x['w'], -x['w']))
#problems.append(problem)

v_l = list(problem["sorted_items"][i]["p"] for i in range(int(problem["n"])))
w_l = list(problem["sorted_items"][i]["w"] for i in range(int(problem["n"])))
#sol_l = list(problem["sorted_items"][i]["x"] for i in range(int(problem["n"])))
file = open( "../knapsackProblemInstances/problemInstances/"+name+"/outp.out", 'r')
line = file.readline()
line = line.replace("\n", "")
problem["z"]=int(line)
max_val = problem["z"]
cap = int(problem["c"])
# max_v = opt_utils.solve_knapsack(v_l, w_l, cap)
start_date = datetime.now()
max_vG, max_cG, sol_l = opt_utils.solve_knapsack_gurobi_multiple(v_l, w_l, cap, maxSoution=2000)
end_date = datetime.now()

print(
    "Problem: " + problem["name"] + " " + str((end_date - start_date).total_seconds()) + "sec; sol: " + str(len(sol_l)))
if (int(max_val) != int(max_vG)):
    print("Sol Error max_val/maxvG: " + str(max_val) + "/" + str(max_vG) + " Gurobi int 9.5.2")
