import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(style="white", color_codes=True)

set_name = sys.argv[1]
model_name = sys.argv[2]
result_name = sys.argv[3]
csv_file = "%s/model-%s-%s-rmse.csv" %(set_name, model_name, result_name)
print(result_name)

rmse_data = pd.read_csv(csv_file)
training_rmse = rmse_data['training']
validation_rmse = rmse_data['validation']

fig, ax = plt.subplots(figsize=(5, 3))

best_epoch = 0
min_rmse = 99999.0
for i in range(len(validation_rmse)):
    if validation_rmse[i] < min_rmse:
        min_rmse = validation_rmse[i]
        best_epoch = i + 1

ax.plot(range(1, len(training_rmse)+1), training_rmse, label='training')
ax.plot(range(1, len(validation_rmse)+1), validation_rmse, label='validation')

ax.vlines(best_epoch, 0, 2, color='r', linestyles='--', zorder=4, label='selected model')

ax.set_xlabel('Epoch')
ax.set_ylabel('RMSE')

ax.set_xlim(0, len(training_rmse)+1)
ax.set_xticks(range(0, len(training_rmse)+1, 2))
ax.grid(True, axis='y')
ax.set_ylim(0.8, 2)
ax.set_yticks(np.arange(0.8, 2, 0.2))

ax.legend(frameon=True, loc='lower left')
fig.tight_layout()
#fig.savefig('%s/%s/%s-rmse.pdf' %(set_name, model_name, result_name))
fig.savefig('%s/model-%s-%s-rmse.pdf' %(set_name, model_name, result_name))
#result_file = "%s/%s/%s-predictions.csv" %(set_name, model_name, result_name)
result_file = "%s/model-%s-%s-predictions.csv" %(set_name, model_name, result_name)
result_data = pd.read_csv(result_file)
for set_n, table in result_data.groupby('set'):
    rmse = ((table['predicted'] - table['real']) ** 2).mean() ** 0.5
    mae = (np.abs(table['predicted'] - table['real'])).mean()
    corr = scipy.stats.pearsonr(table['predicted'], table['real'])
    lr = LinearRegression()
    lr.fit(table[['predicted']], table['real'])
    y_ = lr.predict(table[['predicted']])
    sd = (((table['real'] - y_) ** 2).sum() / (len(table) - 1)) ** 0.5
    print('%10s set: RMSE=%.3f, MAE=%.3f, R=%.2f (p=%.2e), SD=%.3f' % (set_n, rmse, mae, *corr, sd))
    snsplot = sns.jointplot(x='real', y='predicted', data=table, xlim=(0, 15), ylim=(0,15))
    snsplot_path = "%s/model-%s-%s_graph.png"%(set_name, model_name, result_name)
    snsplot.savefig(snsplot_path)
print('\n')
