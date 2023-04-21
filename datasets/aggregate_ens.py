import numpy as np
import pandas as pd

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

ens_cpu = pd.read_csv('../../datasets/T_INSTANCE_CPU.csv')
imc_cpu_p95 = []
imc_cpu_mean = []
imc_cpu_std = []

region_lst = ens_cpu['region_id'].unique().tolist()
for num, i in enumerate(region_lst):
    tmp = ens_cpu.loc[ens_cpu['region_id'] == i].copy()
    tmp_stat = tmp['ifnull(cpu_rate, 0)'].describe([.05, .25, .5, .75, .95]).values.tolist()
    imc_cpu_mean.append(tmp_stat[1])
    imc_cpu_std.append(tmp_stat[2])
    imc_cpu_p95.append(tmp_stat[-2])
    if num % 10 == 0:
        print(f"{num} already")

a1 = np.array(imc_cpu_p95)
a2 = np.array(imc_cpu_mean)
a3 = np.array(imc_cpu_std)
np.save('./imc_cpu_p95.npy', a1)
np.save('./imc_cpu_mean.npy', a2)
np.save('./imc_cpu_std.npy', a3)
