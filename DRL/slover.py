# from ortools.linear_solver import pywraplp
# import numpy as np
#
# # 定义每个函数的参数
# decay_rates = [
#
#     [20, 25, 4],  # f2 的参数
#     [22, 28, 6],  # f3 的参数
#     [23, 29, 7],  # f4 的参数
#     [26, 32, 5],  # f5 的参数
# [24, 31, 5],  # f1 的参数
# ]
#
#
# # 定义函数 f_i
# def f(cpu, io, cache, decay_rate):
#     return (np.exp(-cpu / decay_rate[0]) +
#             np.exp(-io / decay_rate[1]) +
#             np.exp(-cache / decay_rate[2])) * 200
#
#
# # 初始化 OR-Tools 求解器
# solver = pywraplp.Solver.CreateSolver('SCIP')
# if not solver:
#     raise Exception("Solver not created")
#
# # 获取任务数量
# n = len(decay_rates)
#
# # 定义变量
# cpu = [solver.IntVar(1, 50, f'cpu_{i}') for i in range(n)]
#
# io = [solver.NumVar(1, 100, f'io_{i}') for i in range(n)]
# cache = [solver.NumVar(1, 128, f'cache_{i}') for i in range(n)]
#
# # 添加约束条件
# solver.Add(sum(cpu) == 50)
# solver.Add(sum(io) == 100)
# solver.Add(sum(cache) == 128)
#
#
# def computer():
#     total_value = 0
#     for i in range(n):
#         total_value += f(cpu[i], io[i], cache[i], decay_rates[i])
#     return total_value
#
# total_value = computer()
# print(total_value)
#
# solver.Minimize(total_value)
#
# # 求解问题
# status = solver.Solve()
#
# if status == pywraplp.Solver.OPTIMAL:
#     optimized_cpu_io_cache = np.zeros((n, 3))
#     for i in range(n):
#         optimized_cpu_io_cache[i, 0] = cpu[i].solution_value()
#         optimized_cpu_io_cache[i, 1] = io[i].solution_value()
#         optimized_cpu_io_cache[i, 2] = cache[i].solution_value()
#
#     print("Optimized variables:")
#     print(optimized_cpu_io_cache)
#     print("Minimum average function value:")
#     print(np.mean(
#         [f(optimized_cpu_io_cache[i, 0], optimized_cpu_io_cache[i, 1], optimized_cpu_io_cache[i, 2], decay_rates[i]) for
#          i in range(n)]))
# else:
#     print("The problem does not have an optimal solution.")



import numpy as np
from scipy.optimize import minimize


def task_slover(decay_rates):


    # 定义函数 f_i
    def f(cpu, io, cache, decay_rate):
        return (np.exp(-cpu / decay_rate[0]) +
                np.exp(-io / decay_rate[1]) +
                np.exp(-cache / decay_rate[2])) * 200

    # 定义目标函数
    def objective(x):
        total_value = 0
        for i in range(n):
            total_value += f(x[i], x[i+n], x[i+2*n], decay_rates[i])
        return total_value

    # 定义约束条件
    # 定义约束条件
    def constraint(x):
        cpu_sum = np.sum(x[:n])  # 取整
        io_sum = np.sum(x[n:2*n])
        cache_sum = np.sum(x[2*n:])
        return [cpu_sum - 50, io_sum - 100, cache_sum - 128]

    # 获取任务数量
    n = len(decay_rates)

    # 初始猜测的变量值
    x0 = np.random.rand(3*n)

    # 定义约束条件类型
    constraint_type = [{'type': 'eq', 'fun': lambda x: constraint(x)}]
    # 设置变量的取值范围
    bounds = [(1, 50)] * n + [(1, 100)] * n + [(1, 128)] * n

    # 最小化目标函数，同时满足约束条件
    result = minimize(objective, x0, bounds=bounds,constraints=constraint_type)

    # 提取最优解
    optimal_vars = result.x

    # 计算最小平均函数值
    min_avg_value = np.mean([f(optimal_vars[i], optimal_vars[i+n], optimal_vars[i+2*n], decay_rates[i]) for i in range(n)])

    # print("最小平均函数值:", min_avg_value)

    result = optimal_vars.reshape((3,n)).T

    # print("最优解:\n",result )

    prob =  result / np.sum(result, axis=0)

    # print(np.sum(prob, axis=0))

    # x1 = np.random.rand(3*n)
    # x1[:n] = 50/n
    # x1[n:2*n] = 100/n
    # x1[n*2:3*n] = 128/n
    # print( np.mean([f(x1[i], x1[i+n], x1[i+2*n], decay_rates[i]) for i in range(n)]))

    return prob

if __name__ == '__main__':
    # 定义每个函数的参数
    decay_rates = [
        [20, 25, 4],  # f2 的参数
        [40, 28, 8],  # f3 的参数
        [23, 16, 7],  # f4 的参数
        [4, 28, 51],  # f5 的参数
        [34, 54, 3]  # f1 的参数
    ]

    prob = task_slover(decay_rates)