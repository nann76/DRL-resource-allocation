import time

import numpy as np

# 任务数量
NUM_TASK = 3

# 总缓存大小（128GB）
TOTAL_CACHE = 128

# 缓存分配上下界
CACHE_MIN = 0
CACHE_MAX = 128


# 模拟任务延迟函数（假设一个简单的反比关系）
delacy_list = [1,30,40]

def func(x,delacy):

    return np.exp(-x/delacy) *100

def compute_delay(cache_allocation):

    total_declay = 0
    for i in range(NUM_TASK):
        total_declay += func(cache_allocation[i],delacy_list[i])

    return total_declay


# 生成初始解
def initial_solution():
    x0 = [TOTAL_CACHE/NUM_TASK]*NUM_TASK
    return x0


# 邻域生成函数
def get_neighbor(solution,lb,ub,change_precision=0.5):
    '''
    :param solution:
    :param lb: 下限
    :param ub: 上线
    :param change_precision: 调整的精确度
    :return:
    '''
    neighbor = solution.copy()
    # 选择要改变的索引
    i = np.random.randint(len(solution))
    # 要改变的大小
    change = np.random.uniform(-change_precision, change_precision)  # 在-change_precision到+change_precision之间调整
    # 修改，且满足上下限
    neighbor[i] = max(lb, min(ub, neighbor[i] + change))

    # 调整其他任务的缓存大小，以保持总和为TOTAL_CACHE
    remaining_cache = TOTAL_CACHE - np.sum(neighbor)
    remaining_tasks = NUM_TASK - 1
    # 将变化的部分分摊到其为
    allo_remaining_cache = remaining_cache / remaining_tasks
    for j in range(len(solution)):
        if j != i:
            neighbor[j] += allo_remaining_cache

    return neighbor


# 模拟退火算法
def simulated_annealing(func, x0, initial_temp, cooling_rate, max_iter,lb,ub,precision):

    current_solution = x0
    current_delay = func(current_solution)

    best_solution = current_solution
    best_delay = current_delay

    temperature = initial_temp

    for iteration in range(max_iter):
        new_solution = get_neighbor(current_solution,lb,ub,precision)
        new_delay = func(new_solution)
        delta_delay = new_delay - current_delay

        if delta_delay < 0 or np.exp(-delta_delay / temperature) > np.random.rand():
            current_solution = new_solution
            current_delay = new_delay

            if current_delay < best_delay:
                best_solution = current_solution
                best_delay = current_delay

        temperature *= cooling_rate
    print(temperature)
    return best_solution, best_delay

def simulated_annealing2(func, x0, T_max, T_min, L, max_stay_counter, lb, ub):
    '''
    :param func: ：目标函数，即需要最小化的函数。
    :param x0: 初始解向量，表示算法开始时的解。
    :param T_max: 初始温度，控制接受次优解的概率。
    :param T_min: 最低温度，当温度降低到此值时停止算法。
    :param L: 总迭代次数
    :param max_stay_counter: 在达到最低温度后，如果连续max_stay_counter次迭代中最优解没有改变，则停止算法
    :param lb: 解向量的下界，即解向量中每个元素的最小值。
    :param ub: 解向量的上界，即解向量中每个元素的最大值
    :return:
    '''
    current_solution = x0  # 当前解
    current_delay = func(current_solution)  # 当前解的延迟
    best_solution = current_solution  # 最优解
    best_delay = current_delay  # 最优解的延迟

    temperature = T_max  # 初始温度
    stay_counter = 0  # 记录连续迭代中最优解未改变的次数

    for iteration in range(L):
        new_solution = get_neighbor(current_solution,lb,ub)  # 生成邻居解
        new_delay = func(new_solution)  # 邻居解的延迟

        delta_delay = new_delay - current_delay  # 延迟变化量

        # 判断是否接受邻居解
        if delta_delay < 0 or np.exp(-delta_delay / temperature) > np.random.rand():
            current_solution = new_solution
            current_delay = new_delay

            # 更新最优解
            if current_delay < best_delay:
                best_solution = current_solution
                best_delay = current_delay
                stay_counter = 0
            else:
                stay_counter += 1

        temperature = temperature * cooling_rate  # 降低温度

        # 检查是否达到最低温度和连续迭代中最优解未改变的次数
        if temperature <= T_min and stay_counter >= max_stay_counter:
            break
    # print(iteration, temperature)
    return best_solution, best_delay
    '''
    :param func: ：目标函数，即需要最小化的函数。
    :param x0: 初始解向量，表示算法开始时的解。
    :param T_max: 初始温度，控制接受次优解的概率。
    :param T_min: 最低温度，当温度降低到此值时停止算法。
    :param L: 每个温度下的迭代次数
    :param max_stay_counter: 在达到最低温度后，如果连续max_stay_counter次迭代中最优解没有改变，则停止算法
    :param lb: 解向量的下界，即解向量中每个元素的最小值。
    :param ub: 解向量的上界，即解向量中每个元素的最大值
    :return:
    '''
def simulated_annealing3(func, x0, T_max, T_min, L, max_stay_counter,cooling_rate,precision, lb, ub,change_precision):
    '''
    :param func: 目标函数
    :param x0: 初始解向量
    :param T_max: 初始温度
    :param T_min: 最低温度
    :param L: 每个温度下的迭代次数
    :param max_stay_counter: 在达到最低温度后，如果连续max_stay_counter次迭代中最优解没有改变，则停止算法
    :param cooling_rate: 温度变化率
    :param precision: 在达到最低温度后，统计stay_counter的精度
    :param lb: 解向量的下界
    :param ub: 解向量的上界
    :param change_precision: 邻域生成算子的调整精度
    :return:
    '''


    current_solution = x0  # 当前解
    current_delay = func(current_solution)  # 当前解的延迟
    best_solution = current_solution  # 最优解
    best_delay = current_delay  # 最优解的延迟

    temperature = T_max  # 初始温度
    flag_reach_min_temp =False
    stay_counter = 0  # 记录连续迭代中最优解未改变的次数


    while(True):
        for i in range(L):
            new_solution = get_neighbor(current_solution,lb,ub,change_precision)  # 生成邻居解
            new_delay = func(new_solution)  # 邻居解的延迟

            delta_delay = new_delay - current_delay  # 延迟变化量

            # 判断是否接受邻居解
            if delta_delay < 0 :
                current_solution = new_solution
                current_delay = new_delay
                best_solution = current_solution
                best_delay = current_delay


                # 到达最低温，新解有一定的下降，若下降精度小于
                if flag_reach_min_temp :
                    if abs(delta_delay) < precision:
                        stay_counter += 1
                    else:
                        stay_counter = 0

            elif np.exp(-delta_delay / temperature) > np.random.rand() :

                current_solution = new_solution
                current_delay = new_delay
                stay_counter += 1


        if temperature > T_min:
            temperature = temperature * cooling_rate  # 降低温度
        else:
            flag_reach_min_temp = True

        # 检查是否达到最低温度和连续迭代中最优解未改变的次数
        if flag_reach_min_temp and stay_counter >= max_stay_counter:
            break
    # print(iteration, temperature)
    return best_solution, best_delay


# 参数设置
initial_temp = 10000
cooling_rate = 0.995
max_iter = 10000

# 执行模拟退火算法
s = time.process_time()
best_solution, best_delay = simulated_annealing3(func=compute_delay,x0=initial_solution(),
                                                T_max=100, T_min=1e-3, L=100, max_stay_counter=200,precision=0.01,
                                                cooling_rate=0.9,lb=CACHE_MIN,ub=CACHE_MAX,change_precision=0.5)
print("time: ",time.process_time()-s)
print("最优缓存分配：", best_solution)
print("最小总延迟：", best_delay)

x0 =initial_solution()
s = time.process_time()
best_solution, best_delay = simulated_annealing(compute_delay,x0, initial_temp, cooling_rate, max_iter,lb=CACHE_MIN,ub=CACHE_MAX,precision=0.5)
print("time: ",time.process_time()-s)
print("最优缓存分配：", best_solution)
print("最小总延迟：", best_delay)

print('均分',[TOTAL_CACHE/NUM_TASK]*NUM_TASK,compute_delay([TOTAL_CACHE/NUM_TASK]*NUM_TASK))
