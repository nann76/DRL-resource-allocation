import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import random_split
import os
from predict_model.pre_model import Pre_MLP

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(50)

class Env:

    def __init__(self,num_tasks=3,num_states=3,
                 train_batch_size=4096,validate_batch_size=100):

        self.num_tasks =num_tasks
        self.max_num_tasks = 5
        self.num_states = num_states



        # 环境模型
        self.model_list = []

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device( "cpu")




        tasks_list = np.arange(self.max_num_tasks)
        states_list = np.arange(self.num_states)
        grid1, grid2 = np.meshgrid(tasks_list, states_list, indexing='ij')
        # [task_id,state_id] 5*3
        self.task_state = np.column_stack((grid1.ravel(), grid2.ravel()))

        # [3^5,5] 5个task的所有state排列
        self.all_state = np.vstack(np.meshgrid(states_list, states_list, states_list, states_list, states_list)).reshape(self.max_num_tasks, -1).T

        self.decay = self.all_decay()
        self.all_label = self.gen_label()
        # self.all_label = np.load('label_5.npy')

        self.all_sample = self.gen_total_instances()

        print(1)




    # def gen_total_instances(self, num_task):
    #     '''
    #     生成所有任务对应曲线的曲线采样向量，共num_task * num_state
    #     :param num_task:
    #     :return:
    #     '''
    #
    #
    #     # list_task = list(range(num_task))
    #     num_sample = 128
    #     zero = torch.zeros((num_sample,)).unsqueeze(-1)
    #     cpu = torch.linspace(0, 1, num_sample) * 50
    #     io = torch.linspace(0, 1, num_sample) * 100
    #     cache = torch.linspace(0, 1, num_sample) * 128
    #
    #     cpu = torch.cat((cpu.unsqueeze(-1),zero,zero),dim=-1)
    #     io = torch.cat((zero, io.unsqueeze(-1),zero,), dim=-1)
    #     cache = torch.cat((zero, zero, cache.unsqueeze(-1)), dim=-1)
    #
    #     state = torch.tensor([0,1,2]).unsqueeze(-1).repeat(1,1,num_sample).squeeze()
    #
    #     # [num_sample, 4]
    #     s1 = torch.cat((state[0].unsqueeze(-1),cpu),dim=-1)
    #     s2 = torch.cat((state[1].unsqueeze(-1),cpu),dim=-1)
    #     s3 = torch.cat((state[2].unsqueeze(-1),cpu),dim=-1)
    #     # [3*num_sample, 4]
    #     state_cpu = torch.cat((s1,s2,s3),dim=0).view(-1,num_sample,4)
    #
    #     # [num_sample, 4]
    #     s1 = torch.cat((state[0].unsqueeze(-1), io), dim=-1)
    #     s2 = torch.cat((state[1].unsqueeze(-1), io), dim=-1)
    #     s3 = torch.cat((state[2].unsqueeze(-1), io), dim=-1)
    #     # [3*num_sample, 4]
    #     state_io = torch.cat((s1, s2, s3), dim=0).view(-1,num_sample,4)
    #
    #     # [num_sample, 4]
    #     s1 = torch.cat((state[0].unsqueeze(-1), cache), dim=-1)
    #     s2 = torch.cat((state[1].unsqueeze(-1), cache), dim=-1)
    #     s3 = torch.cat((state[2].unsqueeze(-1), cache), dim=-1)
    #     # [3*num_sample, 4]
    #     state_cache = torch.cat((s1, s2, s3), dim=0).view(-1,num_sample,4)
    #
    #
    #     y_list = []
    #     index_list = []
    #
    #
    #
    #     for task_idx in range(num_task):
    #
    #         y_cpu = self.model_list[task_idx](state_cpu)
    #         y_io = self.model_list[task_idx](state_io)
    #         y_cache = self.model_list[task_idx](state_cache)
    #         # [3,128,3] 状态，128个采样点，3资源
    #         c_i_c = torch.cat((y_cpu,y_io,y_cache),dim=-1)
    #         # [3,3,3] 状态，3资源，128个采样点
    #         c_i_c = c_i_c.transpose(-1,-2)
    #         y_list.append(c_i_c)
    #
    #     # [num_task,num_state=3,num_res=3,num_sample=128]
    #     y = torch.stack(y_list, dim=0)
    #
    #     # [num_task,num_state,2]
    #     index = torch.stack(index_list, dim=0)
    #
    #
    #     # 找到曲线采样矩阵中的最小值和最大值
    #     min_val = torch.min(y)
    #     max_val = torch.max(y)
    #     # 缩放矩阵到[0,1]范围内
    #     y = (y - min_val) / (max_val - min_val)
    #
    #     return y, index

    def gen_total_instances(self,num_task=5):
        '''
        nnennnn
        :param num_task:
        :return:
        '''

        # x [128,3]
        cpu,io,cache = self.x()

        # 根据 all_state 求解

        instances = []
        for i in range(len(self.all_state)):

            # [5,3]
            decay = torch.tensor([ self.decay[t,self.all_state[i,t]]  for t in range(self.max_num_tasks) ])
            # [5, 128]
            y_cpu = self.compute(cpu,decay)

            y_io = self.compute(io, decay)

            y_cache = self.compute(cache, decay)

            # [5,3, 128] task,state,num_sample
            y = torch.stack([y_cpu, y_io, y_cache], dim=1)

            min_value = y.min()
            max_value = y.max()
            # 进行线性变换和归一化
            normalized_y = (y - min_value) / (max_value - min_value)
            instances.append(normalized_y)
        # [243,5,3, 128]
        instances=torch.stack(instances)

        return  instances


    def compute(self,input,decay):
        out = (torch.exp( - input[...,0]/ decay[:,0, None])+
         torch.exp( - input[...,1]/ decay[:,1, None])+
         torch.exp( - input[...,2]/ decay[:,2, None])) *200

        return out

    def x(self):
        num_sample = 128
        # zero = torch.zeros((num_sample,)).unsqueeze(-1)
        # 随机常数
        zero = torch.full((num_sample,),random.random()).unsqueeze(-1)
        cpu = torch.linspace(0, 1, num_sample) * 50
        io = torch.linspace(0, 1, num_sample) * 100
        cache = torch.linspace(0, 1, num_sample) * 128

        # [128,3]
        cpu = torch.cat((cpu.unsqueeze(-1), zero, zero), dim=-1)
        io = torch.cat((zero, io.unsqueeze(-1), zero,), dim=-1)
        cache = torch.cat((zero, zero, cache.unsqueeze(-1)), dim=-1)
        # [3,128,3]
        # cpu_io_cache = torch.stack((cpu, io, cache))

        return cpu,io,cache





    def gen_label(self):
        from slover import task_slover

        all_state = self.all_state

        all_label = []
        for i in range(len(all_state)):

            decay = np.array([ self.decay[t,all_state[i,t]]  for t in range(self.max_num_tasks) ])
            prob = task_slover(decay)
            all_label.append(prob)

        all_label = np.array(all_label)
        np.save(f'label_{self.max_num_tasks}.npy', all_label)

    def all_decay(self):

        decay = []
        for task in range(self.max_num_tasks):

            decay_rate = []
            # k = []
            for state in range(self.num_states):
                temp_decay_rate = []
                # temp_k =[]
                for x in range(3):
                    # 随机生成一个下降速率 [0,1)
                    decay_rate_ = random.random() * 50
                    temp_decay_rate.append(decay_rate_)
                    # k_ = random.random() + 0.5
                    # k_ = 1.
                    # temp_k.append(k_)
                decay_rate.append(temp_decay_rate)
                # k.append(temp_k)
            decay.append(decay_rate)

        return np.array(decay)









if __name__ == '__main__':

    env = Env()


    # env.gen_label()
    env.gen_total_instances2()




