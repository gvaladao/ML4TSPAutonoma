##For all files in a path
## Read problem, i
## check solution with gurobi (check for multiple solutions and if any flag!)
#: normalize, check again with the un-normalized solutions; if any flag!
# possible error of order 2^-53. can br compensated using fractions.
# compute distances  dp(i,j), between problem i and problem j, j one of the other problems
# same for ds(i,j) distances between solutions; for more solutions we take the minimum
# same for number of solutions.
import csv
from datetime import datetime
from os import listdir
import opt_utils


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    for filename in filenames:
        if filename.endswith(suffix):
            yield filename


path = "hardinstances_pisinger/"

problems=[]   #a list of problems
problem_index = 0  # problem index
for f in find_csv_filenames(path):
    file = open(path + f, 'r')
    problem = {"items":[]}  # a dictionary with various propertie + items a list of discts.

    '''
    n 20: number of items
    c 970: capacity
    z 1428: maxValue
    time 0.00

    items: 1 p[1] w[1] x[1]
    ...
    '''
    line_no=0;
    while True:
        line = file.readline()
        if not line:
            break
        #print(line)

        line = line.replace("\n", "")
        if line == "-----":
            line_no = -1
            line = file.readline()
            problems.append(problem)
            start_date = datetime.now()
            v_l=list(problem["items"][i]["p"] for i in range(len(problem["items"])) )
            w_l=list(problem["items"][i]["p"] for i in range(len(problem["items"])) )
            cap=int(problem["c"])
            #max_v = opt_utils.solve_knapsack(v_l, w_l, cap)
            max_vG, max_cG, sol_l = opt_utils.solve_knapsack_gurobi_multiple(v_l, w_l, cap)
            end_date = datetime.now()

            print("Problem: "+problem["name"]+" "+str((end_date-start_date).total_seconds())+"sec; sol: "+str(len(sol_l)))
            text_solver = "*"
            '''if (max_v != max_vG):
                text_solver = "Sol Error max_v/maxvG: " + str(max_v) + "/" + str(max_vG) + "Gurobi int 9.5.2"
                #diff_gurobi_solver += 1
            else:
                text_solver = "Gurobi int 9.5.2"
            '''
            problem = {"items": []}  # a dictionary with various propertie + items a list of discts.

        elif line_no==0 :
            problem["name"]=line;
        elif 1<=line_no <= 4:
            content=line.split(" ")
            problem[content[0]] =  content[1]
        else:
            content = line.split(",")
            item = {"id_x": int(content[0]), "p": int(content[1]), "w": int(content[2]), "x": int(content[3])}
            problem["items"].append(item)
        line_no += 1
    file.close()
    problems.append(problem)
    problem_index += 1
print (problems)