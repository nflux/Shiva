from os import path
import numpy as np

basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "pt_logs_2000/log_obs_right_2.csv"))
f = open(filepath, "r")

file_list = []
for i,line in enumerate(f):
    if i == 0 or i == 1:
        continue
    file_list.append(line.split(','))

set_of_indices = set()
file_np = np.asarray(file_list)
for f in range(0,len(file_np)-1,2):
    # print(f, end=' ')
    for i in range(len(file_np[f])):
        if abs(float(file_np[f,i]) - float(file_np[f+1,i])) > 0.1e-3:
            # print(i-1, end=' ')
            set_of_indices.add(i)
            # print(f+1, i+1)
        # if (float(file_np[f,i]) < 0 and float(file_np[f+1,i]) >= 0) or (float(file_np[f,i]) >= 0 and float(file_np[f+1,i]) < 0):
        #     print(i-1, end=' ')
        #     set_of_indices.add(i-1)
    # print('\n--------------')
    # if f == 10:
    #     break

print(set_of_indices)