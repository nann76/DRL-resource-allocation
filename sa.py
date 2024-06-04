import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sko.SA import SA
from sko.GA import GA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_task =3
total_cache = 128

delacy_list = [1,30,40]

def func(x,delacy):

    return np.exp(-x/delacy) *100

def obj_func(allocation):

    total_declay = 0
    for i in range(num_task):
        total_declay += func(allocation[i],delacy_list[i])

    return total_declay

def constraint_func(allocation):
    '''
    约束函数 资源分配总和
    :param allocation:
    :return:
    '''

    return total_cache - sum(allocation)

constraint_eq = [
    lambda x: total_cache - sum(x)
]

# # 初始化SA算法
# lb = [0] * num_task
# ub = [128] * num_task
#
# ga = GA(func=obj_func, n_dim=num_task, max_iter=50000, lb=lb, ub=ub,
#         constraint_eq=constraint_eq,
#         precision=0.5)
# # sa.to(device=device)
#
# start_time = time.time()
# # 运行SA算法
# best_solution, best_latency = ga.run()
# print(time.time() - start_time)
# print("最佳分配方案：", best_solution)
# print("最小延迟：", best_latency)
#
# print('均分',[total_cache/num_task]*num_task,obj_func([total_cache/num_task]*num_task))
#
#
# Y_history = pd.DataFrame(ga.all_history_Y)
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
# Y_history.min(axis=1).cummin().plot(kind='line')
# plt.show()

# 初始化SA算法
lb = [0] * num_task
ub = [128] * num_task
x0 = [total_cache/num_task]*num_task
sa = SA(func=obj_func, x0=x0, T_max=100, T_min=1e-3, L=100, max_stay_counter=200,
        lb=lb,ub=ub)


# sa.to(device=device)

start_time = time.time()
# 运行SA算法
best_solution, best_latency = sa.run()
print(time.time() - start_time)
print("最佳分配方案：", best_solution)
print("最小延迟：", best_latency)