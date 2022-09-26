import pandas as pd
# data_set="./DataSet1/nitems_30_ninstances_90000_dfresult.pkl"
# data_set="./nitems_60_ninstances_90000_dfresult.pkl"
data_set="./nitems_30_ordered_instances_90000_dfresult.pkl"

dfr = pd.read_pickle(data_set)
# print(len(dfr.Variables[0]))
# print(dfr.Variables[0][0])
examples: int = len(dfr.Variables)
problem_size: int = len(dfr.Variables[0])
statistics_on_solutions = [0] * problem_size
statistics_on_patterns={}
for i in range(examples):
    solution_items: int = int(sum(dfr.Variables[i]))
    statistics_on_solutions[solution_items] += 1
    solution_pattern = ''.join([str(int(dfr.Variables[i][j])) for j in range(problem_size)])
    if solution_pattern in statistics_on_patterns.keys():
        statistics_on_patterns[solution_pattern] += 1
    else:
        statistics_on_patterns[solution_pattern] = 1
print("Statistics on: ",data_set,"\n Number of items / Number of generated problems")
expected_items=0;
for j in range(problem_size):
    print(j+1,": ",statistics_on_solutions[j])
    expected_items += (j+1)*statistics_on_solutions[j]/examples
print("(aprox.) Expected number of items in the solution: ", expected_items)
print("Number of patterns: ",len(statistics_on_patterns.keys()))
