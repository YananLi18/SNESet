import numpy as np
import pandas as pd

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

ens_cpu = pd.read_csv('../../datasets/machine_usage.csv',
                      names=['machine_id','time_stamp', 'cpu_util_percent',
                      'mem_util_percent', 'mem_gps', 'mkpi', 'net_in', 'net_out', 'disk_io_percent'])
imc_cpu_p95 = []
imc_cpu_mean = []
imc_cpu_std = []

imc_mem_p95 = []
imc_mem_mean = []
imc_mem_std = []

region_lst = ens_cpu['machine_id'].unique().tolist()
for num, i in enumerate(region_lst):
    tmp = ens_cpu.loc[ens_cpu['machine_id'] == i].copy()
    tmp_cpu_stat = tmp['cpu_util_percent'].describe([.05, .25, .5, .75, .95]).values.tolist()
    tmp_mem_stat = tmp['mem_util_percent'].describe([.05, .25, .5, .75, .95]).values.tolist()
    imc_cpu_mean.append(tmp_cpu_stat[1])
    imc_cpu_std.append(tmp_cpu_stat[2])
    imc_cpu_p95.append(tmp_cpu_stat[-2])

    imc_mem_mean.append(tmp_mem_stat[1])
    imc_mem_std.append(tmp_mem_stat[2])
    imc_mem_p95.append(tmp_mem_stat[-2])
    if num % 10 == 0:
        print(f"{num} already")

a1 = np.array(imc_cpu_p95)
a2 = np.array(imc_cpu_mean)
a3 = np.array(imc_cpu_std)
np.save('./ecs_cpu_p95.npy', a1)
np.save('./ecs_cpu_mean.npy', a2)
np.save('./ecs_cpu_std.npy', a3)

b1 = np.array(imc_mem_p95)
b2 = np.array(imc_mem_mean)
b3 = np.array(imc_mem_std)
np.save('./ecs_mem_p95.npy', b1)
np.save('./ecs_mem_mean.npy', b2)
np.save('./ecs_mem_std.npy', b3)
