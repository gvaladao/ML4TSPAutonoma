##For all files in a path
## Read problem, i
#? normalize, check again with the un-normalized solutions; if any difference flag!
#? possible error of order 2^-53. can br compensated using fractions.
#!!!!Important!!!
## Sort items by density and if equals by weigth (prices should be proportionals)

# Check if the solution of Pisinger is among ours !
# take Guroby first solution with a timeout of 300 sec.
# compare with Gurobi multiple solutions (with a limit of 10 sec; just note the problems with differences)
#: compute distances  dp(i,j), between problem i and problem j, j one of the other problems
# same for ds(i,j) distances between solutions; for more solutions we take the average
# same for number of solutions.


import csv
from datetime import datetime
from os import listdir
import opt_utils


def find_csv_filenames(path_to_dir, suffix=".csv"):
    filenames = listdir(path_to_dir)
    # for filename in filenames:
    #     if filename.endswith(suffix):
    #         yield filename
    yield "knapPI_12_50_1000.csv"

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
            problem["sorted_items"]=sorted(problem["items"], key=lambda x: (x['p']/x['w'],x['w']))
            problems.append(problem)


            v_l=list(problem["sorted_items"][i]["p"] for i in range(int(problem["n"]) ))
            w_l=list(problem["sorted_items"][i]["w"] for i in range(int(problem["n"])))
            sol_l=list(problem["sorted_items"][i]["x"] for i in range(int(problem["n"])))
            max_val=problem["z"]
            cap=int(problem["c"])
            #max_v = opt_utils.solve_knapsack(v_l, w_l, cap)
            start_date = datetime.now()
            max_vG, max_cG, sol_l = opt_utils.solve_knapsack_gurobi_multiple(v_l, w_l, cap,maxSoution=2000)
            end_date = datetime.now()

            print("Problem: "+problem["name"]+" "+str((end_date-start_date).total_seconds())+"sec; sol: "+str(len(sol_l)))
            if (int(max_val) != int(max_vG)):
                print("Sol Error max_val/maxvG: " + str(max_val) + "/" + str(max_vG) + " Gurobi int 9.5.2")
            '''
            if (max_val != max_vG):
                print("Sol Error max_val/maxvG: " + str(max_val) + "/" + str(max_vG) + " Gurobi int 9.5.2"
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

'''
Academic license - for non-commercial use only - expires 2024-03-02
Problem: knapPI_12_50_1000_1 0.884228sec; sol: 53
Problem: knapPI_12_50_1000_2 12.37881sec; sol: 244
Problem: knapPI_12_50_1000_3 598.516546sec; sol: 2000
Problem: knapPI_12_50_1000_4 488.62405sec; sol: 2000
Problem: knapPI_12_50_1000_5 5.860757sec; sol: 179
Problem: knapPI_12_50_1000_6 554.81126sec; sol: 1704
Problem: knapPI_12_50_1000_7 602.787295sec; sol: 2000
Problem: knapPI_12_50_1000_8 1827.834159sec; sol: 2000
Problem: knapPI_12_50_1000_9 834.841247sec; sol: 2000
Problem: knapPI_12_50_1000_10 74870.56127sec; sol: 2000
Problem: knapPI_12_50_1000_11 1743.29906sec; sol: 24
Problem: knapPI_12_50_1000_12 11042.075518sec; sol: 2000
Problem: knapPI_12_50_1000_13 403.959131sec; sol: 2000
Problem: knapPI_12_50_1000_14 122.946875sec; sol: 114
Problem: knapPI_12_50_1000_15 5314.967325sec; sol: 2000
Problem: knapPI_12_50_1000_16 219.707758sec; sol: 2000
Problem: knapPI_12_50_1000_17 2663.97836sec; sol: 51
Problem: knapPI_12_50_1000_18 562.886593sec; sol: 2000
Problem: knapPI_12_50_1000_19 20664.455467sec; sol: 197
Problem: knapPI_12_50_1000_20 301.157347sec; sol: 2000
Problem: knapPI_12_50_1000_21 4055.381589sec; sol: 2000
Problem: knapPI_12_50_1000_22 38721.116179sec; sol: 2000
Problem: knapPI_12_50_1000_23 77754.806379sec; sol: 636
Problem: knapPI_12_50_1000_24 2437.600896sec; sol: 2000
Problem: knapPI_12_50_1000_25 4068.094244sec; sol: 2000
Problem: knapPI_12_50_1000_26 203.936672sec; sol: 2000
Problem: knapPI_12_50_1000_27 0.472925sec; sol: 40
Problem: knapPI_12_50_1000_28 5730.030546sec; sol: 52
Problem: knapPI_12_50_1000_29 33342.876459sec; sol: 211
Problem: knapPI_12_50_1000_30 812.378895sec; sol: 2000
Problem: knapPI_12_50_1000_31 382.033416sec; sol: 2000
Problem: knapPI_12_50_1000_32 208.909276sec; sol: 2000
Problem: knapPI_12_50_1000_33 919.949434sec; sol: 2000
Problem: knapPI_12_50_1000_34 711.289662sec; sol: 2000
Problem: knapPI_12_50_1000_35 9338.317041sec; sol: 2000
Problem: knapPI_12_50_1000_36 245.042351sec; sol: 2000
Problem: knapPI_12_50_1000_37 681.200025sec; sol: 19
Problem: knapPI_12_50_1000_38 24174.767154sec; sol: 2000
Problem: knapPI_12_50_1000_39 606.164844sec; sol: 5
Problem: knapPI_12_50_1000_40 514.891476sec; sol: 2000
Problem: knapPI_12_50_1000_41 4044.55097sec; sol: 83
Problem: knapPI_12_50_1000_42 954.459521sec; sol: 2000'''