import pyodbc
'''
problem:  94
weights:  [52, 18, 91, 70, 25, 72, 87, 13, 38, 15, 72, 99, 32, 95, 89, 33, 28, 56, 12, 46, 23, 85, 60, 23, 61, 28, 70, 19, 25, 46]
prices:  [39, 9, 43, 12, 18, 47, 32, 37, 30, 8, 38, 21, 43, 36, 6, 8, 2, 31, 20, 39, 44, 32, 26, 12, 34, 15, 8, 26, 18, 27]
capacity:  272
Found 3 optimal feasible Solutions!
Max val:  296.0
1 [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
2 [1.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 1.0, 1.0, -0.0, -0.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 1.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 1.0, 1.0, -0.0]
3 [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
overal solution:  [1.0, 0.0, 0.0, 0.0, 0.6666666666666666, 0.0, 0.0, 1.0, 0.6666666666666666, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.3333333333333333, 0.0, 0.0, 0.0, 1.0, 0.6666666666666666, 0.0]
'''
conn = pyodbc.connect(
    r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=C:\Users\adediu\Documents\knapSack.accdb;')
cursor = conn.cursor()
cursor.execute('select * from items')

for row in cursor.fetchall():
    print(row)