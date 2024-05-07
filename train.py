import copy
import os
import pandas as pd
import torch
import random
import time
import numpy as np
from collections import deque

import config
from environment import  Env
from  agent2 import Agent
from torch.utils.data import DataLoader, TensorDataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(300)


class TrainManager:

    def __init__(self):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')

        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else :
            torch.set_default_tensor_type('torch.FloatTensor')
            num_cpus = torch.get_num_threads()
            print(f"PyTorch默认使用的cPU核个数为：{num_cpus}")
        # print("Pytorch device: ",device.type)
        print('Using PyTorch version:', torch.__version__, ' Device:', device)
        # 设置打印选项  threshold: 控制打印的张量元素数量的阈值， np.inf 表示无限制，可以打印所有元素
        # torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
        #                        sci_mode=False)




    def train(self):


        max_iterations =  1000
        policy_update_timestep =  1
        validate_timestep =  10      #每次验证，同时检查此次模型是否优于上一次的，如果优于则保存



        maxlen_best_model = 2  # Save the best model
        models_best = [None for _ in range(maxlen_best_model)]
        makespan_best = [float('inf') for _ in range(maxlen_best_model)]



        agent = Agent()

        env = Env()

        start_train_time = time.time()
        for i in range(1,max_iterations + 1):
            print(f'iter: {i}')

            state, index = env.train_dataset

            # 合并state和index_tensor为一个数据集
            combined_dataset = TensorDataset(state, index)
            # 使用DataLoader从合并后的数据集中采样
            dataloader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
            # 遍历dataloader获取每次的采样
            for batch_data in dataloader:
                sampled_state, sampled_index = batch_data

                action_probs = agent.get_action(sampled_state)
                _, reward, _ = env.step(action_probs, sampled_index)

                # 模型更新
                agent.learn( reward, action_probs)


                # 验证集验证
                val_state, val_index = env.validate_dataset
                action_probs = agent.get_action(val_state)
                val_reward = env.complete_delay(action_probs, val_index)
                if torch.isnan(torch.mean(val_reward)):
                    print("mean(val_reward) contains NaN")
                print('validate mean delay: ',torch.mean(val_reward).item())




if __name__ == '__main__':

    t = TrainManager()
    t.train()
