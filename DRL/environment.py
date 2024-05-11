import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import random_split

class Env:

    def __init__(self,num_tasks=6,num_states=5,
                 train_batch_size=4096,validate_batch_size=100):

        self.num_tasks =num_tasks
        self.max_num_tasks = 20
        self.num_states = num_states

        self.train_batch_size = train_batch_size


        # [num_tasks,num_states] 所有任务所有状态的曲线下降速率
        self.list_decay_rate = []

        # 生成所有曲线
        for task in range(self.max_num_tasks):
            states_decay_rate = []
            for state in range(self.num_states):
                # 随机生成一个下降速率
                decay_rate = torch.rand(1) * 500
                states_decay_rate.append(decay_rate)

                # decay_rate = np.random.uniform(0.5, 5)  # 随机生成下降速率范围
                # offset = np.random.uniform(0, 50)  # 随机生成偏移量范围
                # y = torch.exp(-(x + offset) / decay_rate) * 128
            self.list_decay_rate.append(states_decay_rate)


        self.all_instances = self.gen_total_instances(num_task=self.max_num_tasks)

        # batch_dataset_state,batch_dataset_index = self.gen_instances(size_dataset=3000,  num_task=num_tasks)
        self.train_dataset = self.gen_instances(size_dataset=train_batch_size, num_task=num_tasks)

        self.validate_dataset = self.gen_instances(size_dataset=validate_batch_size, num_task=num_tasks)

    def update_train_dataset(self):
        self.train_dataset = self.gen_instances(size_dataset=self.train_batch_size, num_task=self.num_tasks)

    def sample_x(self,num_sample):
        '''
        采样x，均匀采样
        :param num_sample:
        :return:
        '''

        # 在[0, 128]范围内随机采样n个浮点数
        # random_samples = torch.rand(num_sample) * 128

        random_samples = torch.linspace(0, 1, num_sample) * 128
        sorted_samples_x, indices = torch.sort(random_samples)
        # 返回采样的x
        return sorted_samples_x



    def gen_instances(self,size_dataset,num_task):
        '''
        生成 size_dataset 个 num_task 个任务的数据集
        :param size_dataset:
        :param num_task:
        :return:
        '''

        total_task_list = list(range(self.max_num_tasks))
        total_state_list = list(range(self.num_states))

        dataset_state = []
        task_state_index = []

        for batch in range(size_dataset):
            # 所有任务中随机选择 num_task 个task

            choose_task = random.sample(total_task_list, num_task)

            # choose_task = list(range(num_task))
            # 每个任务随机选择一个状态
            choose_task_states = [random.choice(total_state_list) for _ in range(num_task)]


            state = self.all_instances[0][choose_task,choose_task_states]
            index_tensor = self.all_instances[1][choose_task, choose_task_states]

            dataset_state.append(state)
            task_state_index.append(index_tensor)


        # [batch_size,num_task,128]
        dataset_state = torch.stack(dataset_state, dim=0)
        # [batch_size,num_task,2]
        task_state_index = torch.stack(task_state_index, dim=0)


        return dataset_state,task_state_index

    def gen_total_instances(self, num_task):
        '''
        生成所有任务对应曲线的曲线采样向量，共num_task * num_state
        :param num_task:
        :return:
        '''


        # list_task = list(range(num_task))


        x = self.sample_x(num_sample=128)

        y_list = []
        index_list = []

        for task_idx in range(num_task):

            decay_rate = self.list_decay_rate[task_idx][:]
            decay_rate= torch.cat(decay_rate).view(-1)

            y = self.exponential_decay(x.repeat(self.num_states, 1), decay_rate.unsqueeze(1).repeat(1, 128))
            y_list.append(y)

            index_tensor = torch.cat((torch.tensor([task_idx] * self.num_states).unsqueeze(-1),
                                      torch.tensor(list(range(self.num_states))).unsqueeze(-1)), dim=-1)
            index_list.append(index_tensor)

        # [num_task,num_state,128] -> [num_task*num_state,128]
        # y = torch.stack(y_list, dim=0).view(-1, 128)
        # [num_task,num_state,128]
        y = torch.stack(y_list, dim=0)

        # [num_task,num_state,2]
        index = torch.stack(index_list, dim=0)


        # 找到曲线采样矩阵中的最小值和最大值
        min_val = torch.min(y)
        max_val = torch.max(y)
        # 缩放矩阵到[0,1]范围内
        y = (y - min_val) / (max_val - min_val)

        return y, index

    def exponential_decay(self,x,decay_rate):
        '''
        :param x: input
        :param decay_rate: 下降速率
        :return:
        '''
        return torch.exp(-x / decay_rate) * 1000


    def draw_cure(self):

        x = torch.arange(0, 128)

        for task in range(self.num_tasks):
            for state in range(self.num_states):
                # 计算指数下降曲线
                decay_rate = self.list_decay_rate[task][state]
                y = self.exponential_decay(x,decay_rate)

                plt.plot(x, y, label=f'Task {task + 1}, State {state + 1}')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Exponential Descent Curves for 20 Tasks with 5 States Each')
        plt.legend()
        plt.grid(True)
        plt.show()

    def step(self, action, index):
        '''
        根据action计算返回reward，
        重新给定state
        :param action:
        :return:
        '''
        if action.dim() > 1:
            # [batch_size,num_task]
            batch_size, num_task = action.size()
        else:
            num_task = action.size(0)
            batch_size = 1

        # 共分128G
        allocate_size = action * 128

        # [batch_size,num_task,2]

        # 根据索引获取衰减率

        decay_rate = torch.zeros((batch_size, num_task))
        for batch_idx in range(batch_size):
            for task_idx in range(num_task):
                task_index = index[batch_idx, task_idx, 0]
                state_index = index[batch_idx, task_idx, 1]
                decay_rate[batch_idx, task_idx] = self.list_decay_rate[task_index][state_index]

        # 减少分配空间，曲线上探
        sub_allocate_size = allocate_size - 128 * 0.05
        sub_allocate_size = torch.where(sub_allocate_size<0.,torch.tensor(0.),sub_allocate_size)

        # 增大分配空间，曲线下探
        add_allocate_size = allocate_size + 128 * 0.05
        add_allocate_size = torch.where(add_allocate_size > 128.,torch.tensor(128.), add_allocate_size)

        list_delay = []

        list_sub_allocate_delay = []
        list_add_allocate_delay = []

        for batch_idx in range(batch_size):
            # total_delay = self.exponential_decay(allocate_size[batch_idx], decay_rate[batch_idx]).sum()
            delay = self.exponential_decay(allocate_size[batch_idx], decay_rate[batch_idx])
            sub_allocate_delay = self.exponential_decay(sub_allocate_size[batch_idx], decay_rate[batch_idx])
            add_allocate_delay = self.exponential_decay(add_allocate_size[batch_idx], decay_rate[batch_idx])

            list_delay.append(delay)
            list_sub_allocate_delay.append(sub_allocate_delay)
            list_add_allocate_delay.append(add_allocate_delay)

        # [batch,num_task]
        list_delay = torch.stack(list_delay, dim=0)
        list_sub_allocate_delay = torch.stack(list_sub_allocate_delay, dim=0)
        list_add_allocate_delay = torch.stack(list_add_allocate_delay, dim=0)

        # 减少分配时，增加的延迟
        add_delay = list_sub_allocate_delay - list_delay
        # 增大分配时，减小的延迟
        sub_delay = list_delay - list_add_allocate_delay

        # 如果存在小于0，改为0
        add_delay = torch.where(add_delay < 0., torch.tensor(0.), add_delay)
        sub_delay = torch.where(sub_delay < 0., torch.tensor(0.), sub_delay)

        # n个task的平均延迟
        mean_list_delay = torch.mean(list_delay, dim=-1,keepdim=True)
        # 平均延迟 / 每个任务的延迟 : 相较于平均延迟，任务延迟越小该奖励越大
        reward_delay = mean_list_delay/ (list_delay + 1e-9)

        # 当减小相同分配大小时，相较于平均增大的延迟，任务增加的延迟越大，说明不应该减少该任务的分配大小，该奖励越大
        mean_add_delay = torch.mean(add_delay, dim=-1,keepdim=True)
        reward_add_delay = add_delay / (mean_add_delay + 1e-9)

        # 当增大相同分配大小时，相较于平均减少的延迟，任务减少的延迟越大，说明应该增大该任务的分配大小，该奖励越大
        mean_sub_delay = torch.mean(sub_delay, dim=-1,keepdim=True)
        reward_sub_delay =  sub_delay / (mean_sub_delay + 1e-9)

        # 奖励由三部分组成,当前分配比例下延迟小，如果增大分配大小延迟减少的多，如果减少分配大小延迟增大的多
        reward = (1.0 * reward_delay) + (0.5 * reward_add_delay) + (0.5 * reward_sub_delay)

        reward = (reward - reward.mean()) / (reward.std() + 1e-9)

        # 无下一个状态
        state = None
        # 一次即结束
        done = None
        return state, reward, done

    def complete_delay(self,action,index):
        '''
         计算返回延迟
        :param action:
        :param index:
        :return:
        '''

        if action.dim()>1:
            # [batch_size,num_task]
            batch_size,num_task = action.size()
        else:
            num_task = action.size(0)
            batch_size = 1

        # 共分128G
        allocate_size = action * 128


        # 根据索引获取衰减率
        decay_rate = torch.zeros((batch_size, num_task))
        for batch_idx in range(batch_size):
            for task_idx in range(num_task):
                task_index = index[batch_idx, task_idx, 0]
                state_index = index[batch_idx, task_idx, 1]
                decay_rate[batch_idx, task_idx] = self.list_decay_rate[task_index][state_index]
                # decay_rate = self.list_decay_rate[task_index][state_index]

        list_total_delay = []
        for batch_idx in range(batch_size):
            # total_delay = self.exponential_decay(allocate_size[batch_idx], decay_rate[batch_idx]).sum()
            total_delay = self.exponential_decay(allocate_size[batch_idx], decay_rate[batch_idx])

            list_total_delay.append(total_delay)

        list_total_delay =  torch.stack(list_total_delay,dim=0)

        return list_total_delay



if __name__ == '__main__':

    env = Env()

    # env.gen_instances(batch_size=400,num_task=3)
    # env.choose_states(20)
    env.draw_cure()
    #
    # x = torch.arange(0, 128)
    # decay_rate = env.list_decay_rate[2][1]
    #
    # print(env.exponential_decay(x,decay_rate))

    # for i in range(5):
    #     print(env.sample())



