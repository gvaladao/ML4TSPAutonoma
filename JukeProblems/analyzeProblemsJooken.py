# f=os.listdir("../knapsackProblemInstances/problemInstances/n_400_c_1000000_g_2_f_0.1_eps_0.1_s_100/")
from datetime import datetime
import statistics
import opt_utils
def print_sol(sol_l:list):
    if len(sol_l) > 1:
        average_sol = []
        sol_var = []
        sol_number = len(sol_l)
        for j in range(len(sol_l[0])):
            sum_of_bits = 0
            bit_variance = 0
            for i in range(sol_number):
                if sol_l[i][j] == ord(b'1'):
                    sum_of_bits += 1
            average_sol.append(sum_of_bits / sol_number)
            # now the variance
            for i in range(sol_number):
                c = 1 if sol_l[i][j] == ord(b'1') else 0
                bit_variance += (c - sum_of_bits / sol_number) * (c - sum_of_bits / sol_number)
            sol_var.append(bit_variance / sol_number)
        print("Solution average: ", average_sol)
        print("Solution variance: ", sol_var)
    else:
        print("Sol: ",sol_l)
num_test_problems:int = 4000
gurobi_minus_solutions:int=0
gurobi_plus_solutions:int=0
gurobi_equal_solutions:int=0
for num_problem in range (1,num_test_problems+1):
    name=str(num_problem).zfill(5)
    print(name,": ")
    file = open( "J10/"+name+".txt", 'r')
    '''
    format first no of items, then (id, p w,x), then capacity,sum(p), sum(w)
    400 :n
    id p,w,x
    --------------
    0 64 1 1
    1 60 5 1
    ...
    399 7936 7942 0
    --------------
    1000000 :c
    1003988 :sum(p)
    999999 :sum(w)
    
    
    '''
    line_no = 0;
    c=0
    problem = {"items": []}  # a dictionary with various propertie + items a list of dicts.
    problem["name"]=name
    while True:
        line = file.readline()
        if not line:
            break
        line = line.replace("\n", "")
        if line_no==0:
            content = line.split(" ")
            problem["n"]=int(content[0])
        elif line_no in range(problem["n"]+4,problem["n"]+8):
            content = line.split(" ")
            problem[content[1][1:]]=int(content[0])
        elif line_no in range(3,problem["n"]+3):
            content = line.split(" ")
            item = {"id_x": int(content[0]), "p": int(content[1]), "w": int(content[2]),"x": int(content[3])}
            problem["items"].append(item)
        else:
            pass # do nothing, just comment lines
        line_no += 1
    file.close()
    problem["sorted_items"] = sorted(problem["items"], key=lambda x: (-x['p'] / x['w'], -x['w']))
    '''
    for i in range(len(problem["sorted_items"])):
        if problem["sorted_items"][i]["p"]!=problem["items"][i]["p"]:
            print("error item: ",i)
    '''
    #problems.append(problem)

    #v_l = list(problem["sorted_items"][i]["p"] for i in range(int(problem["n"])))
    #w_l = list(problem["sorted_items"][i]["w"] for i in range(int(problem["n"])))

    v_l = list(problem["items"][i]["p"] for i in range(int(problem["n"])))
    w_l = list(problem["items"][i]["w"] for i in range(int(problem["n"])))

    sol_l = list(problem["sorted_items"][i]["x"] for i in range(int(problem["n"])))
    #file = open( "../knapsackProblemInstances/problemInstances/"+name+"/outp.out", 'r')
    #line = file.readline()
    #line = line.replace("\n", "")
    #problem["z"]=int(line)
    max_val = problem["sum(p)"]
    cap = problem["c"]
    # max_v = opt_utils.solve_knapsack(v_l, w_l, cap)
    start_date = datetime.now()
    max_vG, max_cG, sol_l = opt_utils.solve_knapsack_gurobi_multiple(v_l, w_l, cap, maxSoution=2000)
    end_date = datetime.now()

    '''
    print(
        "Problem: " + problem["name"] + " " + str((end_date - start_date).total_seconds()) + "sec; sol: " + str(len(sol_l)))
    if (int(max_val) != int(max_vG)):
        print("Sol Error max_val/maxvG: " + str(max_val) + "/" + str(max_vG) + " Gurobi int 9.5.2")
    print_sol(sol_l)
    '''
    if (int(max_val) < int(max_vG)):
        gurobi_plus_solutions +=1
    if (int(max_val) == int(max_vG)):
        gurobi_equal_solutions +=1
    if (int(max_val) > int(max_vG)):
        gurobi_minus_solutions +=1
print('Final results. Gurobi less: {0}, Gurobi= {1}, Gurobi better{2} test problems.'.format(gurobi_minus_solutions,gurobi_equal_solutions,gurobi_plus_solutions) )