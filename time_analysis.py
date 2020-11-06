import sys
import statistics
import numpy as np
import pandas as pd

set_name = sys.argv[1]
model_name = sys.argv[2]
result_name = sys.argv[3]

csv_file = "%s/model-%s-%s-time.csv" %(set_name, model_name, result_name)
print(result_name)
time_data = pd.read_csv(csv_file)

training_time = time_data['time']

mean_value = np.mean(training_time)
std_value = np.std(training_time)
print("mean : %f, std : %f" %(mean_value, std_value))

with open(csv_file, 'r') as f:
    lines = f.read().splitlines()

print(result_name)
print("%d epochs" %(len(lines)-1))
