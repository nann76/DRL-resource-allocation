import copy
import os
import pandas as pd
import torch
import random
import time
import numpy as np
from collections import deque

from environment import  Env
from agent import Agent
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


        max_iterations =  200


        maxlen_best_model = 1  # Save the best model
        makespan_best =float('inf')
        last_best_model_path = None
        count = 0

        list_mean_delay = []

        agent = Agent()

        num_tasks = 6
        env = Env(num_tasks=num_tasks)

        str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

        save_dir = f'./train_dir/num_task_{num_tasks}_{str_time}'
        os.makedirs(save_dir)

        start_train_time = time.time()
        for i in range(1,max_iterations + 1):
            print(f'iter: {i}')

            state, index = env.train_dataset

            # 合并state和index_tensor为一个数据集
            combined_dataset = TensorDataset(state, index)
            # 使用DataLoader从合并后的数据集中采样
            dataloader = DataLoader(combined_dataset, batch_size=1000, shuffle=True)
            # 遍历dataloader获取每次的采样
            for batch_data in dataloader:
                sampled_state, sampled_index = batch_data

                action_probs = agent.get_action(sampled_state)
                _, reward, _ = env.step(action_probs, sampled_index)

                # 模型更新
                agent.learn( reward, action_probs)
                count +=1


                # 验证集验证
                val_state, val_index = env.validate_dataset
                action_probs = agent.get_action(val_state)
                val_reward = env.complete_delay(action_probs, val_index)
                # if torch.isnan(torch.mean(val_reward)):
                #     print("mean(val_reward) contains NaN")

                mean_delay = torch.mean(val_reward).item()
                list_mean_delay .append(mean_delay)
                print('validate mean delay: ',mean_delay)

                if mean_delay < makespan_best:

                    makespan_best = mean_delay
                    save_new_model_path = '{0}/model_T{1}_I{2}.pt'.format(save_dir,num_tasks, count)
                    if last_best_model_path != None:
                        os.remove(last_best_model_path)
                    last_best_model_path = save_new_model_path

                    torch.save(agent.model.state_dict(), save_new_model_path)


        seconds = time.time() - start_train_time
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = int(seconds % 60)
        print(f"total_train_time: H:{hours}-M:{minutes}-S:{remaining_seconds}")
        print(list_mean_delay)

        with open('{0}/list.txt'.format(save_dir),'w') as f:

            f.write('valid_list_mean_delay' + str(list_mean_delay) + '\n\n')

        import matplotlib.pyplot as plt

        # plt.switch_backend('Agg')
        plt.figure(figsize=(12, 6))
        x_data = list(range(1,len(list_mean_delay)+1))
        plt.plot( x_data, list_mean_delay, label='mean delay')

        plt.xlabel('iterations')
        plt.ylabel('mean delay')
        plt.legend()

        plt.grid()  # 网格
        plt.tight_layout()  # 去白边
        plt.savefig(save_dir+'/mean_delay.png', dpi=200)
        plt.show()






if __name__ == '__main__':

    t = TrainManager()
    t.train()
