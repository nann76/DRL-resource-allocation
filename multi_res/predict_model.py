import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
import joblib

# https://zhuanlan.zhihu.com/p/551837497

cpu = np.arange(1, 51, 2)  # 0 to 50
io = np.arange(1, 101, 2)  # 0 to 100
cache = np.arange(1, 129, 2)  # 0 to 128
cpu_grid, io_grid, cache_grid = np.meshgrid(cpu, io, cache, indexing='ij')
combinations = np.vstack([cpu_grid.ravel(), io_grid.ravel(), cache_grid.ravel()]).T
def f(cpu_io_cache):
    return (np.exp(-cpu_io_cache[:, 0] / 24) +
            np.exp(-cpu_io_cache[:, 1] / 31) +
            np.exp(-cpu_io_cache[:, 2] / 5)) * 200


X = combinations
y = f(X)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#决策树回归
model_DecisionTreeRegressor = DecisionTreeRegressor()
#SVM回归
model_SVR = svm.SVR()

model = model_DecisionTreeRegressor
# model = model_SVR

# def train(model):
#     model.fit(x_train,y_train)
#     score = model.score(x_test, y_test)
#     result = model.predict(x_test)

# model.fit(x_train,y_train)
# score = model.score(x_test, y_test)
# print(score)

# 交叉验证
scores = cross_val_score(model, X, y, cv=5,n_jobs=-1)
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

model.fit(x_train,y_train)
score = model.score(x_test, y_test)
print(score)
result = model.predict(x_test)
print(result)
print(y_test)

# 模型保存
# joblib.dump(model, 'DecisionTree.pkl')
# 模型加载
# model = joblib.load('DecisionTree.pkl')
import torch
decimal = 4


# def decimal_to_binary_vector(decimal, bit_length=4):
#     """
#     将十进制数转换为定长二进制字符串，并将其转换为特征向量。
#     """
#     bin_str = format(decimal, f'0{bit_length}b')  # 将十进制数转换为二进制字符串
#     binary_vector = [int(bit) for bit in bin_str]  # 将二进制字符串转换为特征向量
#     return np.array(binary_vector)
#
# key_value = decimal_to_binary_vector(4)
# print(key_value)
#
# def state_one_hot(state):
#     state_one_hot = np.zeros((3,))
#     state_one_hot[state] = 1
#
#     return state_one_hot
#
# state = state_one_hot(0)
# print(state)
#
# # 在轴0上拼接数组
# result = np.concatenate((state, key_value), axis=0)
# print(result)