import random
import time

import torch
import numpy as np

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# setup_seed(50)
# class creat_dataset:
#
#     def __init__(self,num_task=5):
#         # 任务数
#         self.num_task = num_task
#         # 每个任务的状态数  workload pattern
#         self.num_state = 3
#         self.list_state = list(range(self.num_state))
#
#         # CPU
#         self.max_num_cpu = 50
#         # 缓存
#         self.max_cache = 128.
#         # I/O 百分比
#         self.max_io = 100.
#
#         self.decay_rate = [[[24.876828437930115, 31.870558072184547, 23.65842899937276], [4.259403250549837, 33.82219767447417, 7.665876428476887], [48.98614263069554, 9.448449339247716, 16.445959727513703]], [[30.374710421587213, 45.76960799368101, 34.657388032942585], [39.691428724892184, 21.51460722302118, 13.309008553163853], [35.003295162388945, 45.54061421852978, 34.623090021753164]], [[13.483181455387527, 28.953775436737317, 31.222094926981097], [40.97202126590042, 31.2731924606579, 24.80524682511288], [40.748813573949874, 6.255457736327591, 25.829561166655253]], [[18.386983408844053, 0.33718249142060674, 29.35056488250476], [48.917386975406814, 41.44011644218936, 11.20920634354401], [4.895987301417715, 18.803767044175707, 1.675186169442272]], [[48.289734443374094, 11.855993674055142, 26.416716509010325], [20.91330355709003, 23.509147796423374, 18.126217295080227], [0.9590856982530982, 31.970935153452622, 15.841084664801652]]]
#
#
#         # self.function = []
#         #
#         # for i in range(self.num_task):
#         #
#         #     func_state = []
#         #     for s in range(self.num_state):
#         #         decay_cpu, decay_io, decay_cache = self.decay_rate[i][s]
#                 f = lambda cpu,io,cache : (torch.exp(-cpu / self.decay_rate[i][s][0]) +
#                                            torch.exp(-io / self.decay_rate[i][s][1])  +
#                                            torch.exp(-cache / self.decay_rate[i][s][2]))
#         #         f = lambda cpu_io_cache: (torch.exp(-cpu_io_cache[ ...,0] / decay_cpu) +
#         #                                     torch.exp(-cpu_io_cache[...,1] / decay_io) +
#         #                                     torch.exp(-cpu_io_cache[...,2] / decay_cache)) * 200
#         #         func_state.append(f)
#         #     self.function.append(func_state)
#         #
#         # print(1)
#         #
#         # t =torch.tensor([[1,2,3],[4,5,6]])
#         # print(self.function[0][0](t))
#
#
#     def computer(self):



if __name__ == '__main__':

    # dataset = creat_dataset()
    # cpu = np.arange(1,51)  # 0 to 50
    # io = np.arange(1,101)  # 0 to 100
    # cache = np.arange(1,129)  # 0 to 128

    cpu = np.arange(1,51,2)  # 0 to 50
    io = np.arange(1,101,2)  # 0 to 100
    cache = np.arange(1,129,2)  # 0 to 128
    cpu_grid, io_grid, cache_grid = np.meshgrid(cpu, io, cache, indexing='ij')
    combinations = np.vstack([cpu_grid.ravel(), io_grid.ravel(), cache_grid.ravel()]).T
    # print(combinations.shape)  # 应该是 (51 * 101 * 129, 3)
    # print(combinations)

    # states = np.arange(3)
    # cpu = np.arange(1, 51, 2)
    # io = np.arange(1, 101, 2)
    # cache = np.arange(1, 129, 2)
    # import itertools
    # combinations = list(itertools.product(states, cpu, io, cache))
    # combinations = np.array(combinations)

    # import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import svm

    from sklearn.metrics import mean_squared_error, r2_score




    def f(cpu_io_cache):
        return (np.exp(-cpu_io_cache[:,0] / 24) +
                np.exp(-cpu_io_cache[:,1] /31) +
                np.exp(-cpu_io_cache[:,2] / 5)) *200

    X= combinations
    y = f(X)


    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建和训练线性回归模型
    # model = LinearRegression()
    model = DecisionTreeRegressor()
    # model = svm.SVR()
    # from thundersvm import SVC
    # model = SVC(kernel='rbf')
    s =time.time()
    model.fit(X_train, y_train)
    print(f'train time : {time.time()-s}')

    # # 获取权重
    # weights = model.coef_
    # intercept = model.intercept_
    #
    # print("Weights:", weights)
    # print("Intercept:", intercept)

    # 预测
    y_pred = model.predict(X_test)

    # print(model.predict(np.array([[1,2,64],[1,2,64.5]])))
    # score = model.score(y_pred, y_test)
    # print(score)

    # plt.figure()
    # plt.plot(np.arange(len(y_pred)), y_test, 'go-', label='true value')
    # plt.plot(np.arange(len(y_pred)), y_pred, 'ro-', label='predict value')
    # # plt.title('score: %f' % score)
    # plt.legend()
    # plt.show()


    # 计算均方误差和R^2得分
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    import joblib
    # 模型保存
    joblib.dump(model, 'DecisionTree.pkl')
    # 模型加载
    # model = joblib.load('DecisionTree.pkl')








