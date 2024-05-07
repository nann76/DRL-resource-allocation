import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


class Env:

    def __init__(self,num_tasks=20,num_states=5,list_task=[]):

        self.num_tasks =num_tasks
        self.max_num_tasks = 20
        self.num_states = num_states

        # 选中的task
        self.list_task = []
        # 此处默认为全选 TODO 之后更改，由于某一时刻可能只是全部task中的几个
        for i in range(self.num_tasks):
            self.list_task.append(i)

        # [num_tasks,num_states] 所有任务所有状态的曲线下降速率
        self.list_decay_rate = []


        for task in range(self.num_tasks):
            states_decay_rate = []
            for state in range(self.num_states):
                # 随机生成一个下降速率
                decay_rate = torch.rand(1) * 500
                states_decay_rate.append(decay_rate)

                # decay_rate = np.random.uniform(0.5, 5)  # 随机生成下降速率范围
                # offset = np.random.uniform(0, 50)  # 随机生成偏移量范围
                # y = torch.exp(-(x + offset) / decay_rate) * 128
            self.list_decay_rate.append(states_decay_rate)


        # self.state = self.gen_state()
        self.train_dataset = self.gen_total_instances(num_task=num_tasks)

        # self.train_dataset = self.gen_instances(size_dataset=2000,num_task=num_tasks)
        # self.validate_dataset = self.gen_instances(size_dataset=200, num_task=num_tasks)

        self.validate_dataset = copy.deepcopy(self.train_dataset)


    def sample_x(self,num_sample):
        '''
        采样x
        :param num_sample:
        :return:
        '''

        # 在[0, 128]范围内随机采样n个浮点数
        # random_samples = torch.rand(num_sample) * 128

        random_samples = torch.linspace(0, 1, num_sample) * 128
        sorted_samples_x, indices = torch.sort(random_samples)
        # 返回采样的x
        return sorted_samples_x

    def choose_states(self,num_tasks):
        '''
        随机选择每个task中的一个状态
        :param num_tasks: 要生成的个数
        :return:
        '''

        state_indices = torch.randint(0, self.num_states, (num_tasks,))
        # 将类别索引转换为独热码 [num_tasks,num_states]
        one_hot_matrix_states = F.one_hot(state_indices, num_classes=self.num_states)
        # print(one_hot_matrix_states)

        return state_indices,one_hot_matrix_states

    def gen_state(self,list_task,num_tasks):



        # [num_tasks,num_tasks]
        self.one_hot_matrix_tasks = F.one_hot(torch.tensor(list_task), num_classes=self.max_num_tasks)
        # self.one_hot_matrix_tasks = torch.eye(self.num_tasks)

        # 每个task选择的state，及其one-hot
        # [num_tasks,num_states] 每个task从5个状态中随机选一个
        state_indices, one_hot_matrix_states = self.choose_states(num_tasks=num_tasks)

        # x采样，计算对应y
        x = self.sample_x(num_sample = 64-self.max_num_tasks-self.num_states)
        y_list = []
        for task in self.list_task:
            # 挑选task对应state的曲线，并根据采样的x计算y
            decay_rate = self.list_decay_rate[task][state_indices[task]]
            y = self.exponential_decay(x,decay_rate).unsqueeze(0)
            y_list.append(y)

        y = torch.cat(y_list,dim=0)

        # 找到矩阵中的最小值和最大值
        min_val = torch.min(y)
        max_val = torch.max(y)
        # 缩放矩阵到[0,1]范围内
        scaled_y = (y - min_val) / (max_val - min_val)

        state = torch.cat((self.one_hot_matrix_tasks,one_hot_matrix_states,scaled_y),dim=1)

        # [num_task,2] 分别为task和state的索引
        index_tensor = torch.cat((torch.tensor(list_task).unsqueeze(-1),state_indices.unsqueeze(-1)),dim=-1)




        return state,index_tensor


    def gen_state2(self,list_task,num_tasks):
        '''
        纯粹曲线采样
        :return:
        '''


        # 每个task选择的state，及其one-hot
        # [num_tasks,num_states] 每个task从5个状态中随机选一个
        state_indices, one_hot_matrix_states = self.choose_states(num_tasks=num_tasks)

        # x采样，计算对应y
        x = self.sample_x(num_sample=128)

        y_list = []
        for task in self.list_task:
            # 挑选task对应state的曲线，并根据采样的x计算y
            decay_rate = self.list_decay_rate[task][state_indices[task]]
            y = self.exponential_decay(x, decay_rate).unsqueeze(0)
            y_list.append(y)

        y = torch.cat(y_list, dim=0)

        # 找到矩阵中的最小值和最大值
        min_val = torch.min(y)
        max_val = torch.max(y)
        # 缩放矩阵到[0,1]范围内
        scaled_y = (y - min_val) / (max_val - min_val)

        state = scaled_y
        # [num_task,2] 分别为task和state的索引
        index_tensor = torch.cat((torch.tensor(list_task).unsqueeze(-1), state_indices.unsqueeze(-1)), dim=-1)

        return state, index_tensor



    def gen_instances(self,size_dataset,num_task):
        # TODO 默认前num_task个task

        list_task = []
        for i in range(num_task):
            list_task.append(i)

        dataset_state = []
        task_state_index = []

        for batch in range(size_dataset):
            state,index_tensor = self.gen_state2(list_task,num_tasks=num_task)
            dataset_state.append(state)
            task_state_index.append(index_tensor)

        # [batch_size,num_task,64]
        dataset_state = torch.stack(dataset_state, dim=0)
        # [batch_size,num_task,2]
        task_state_index = torch.stack(task_state_index, dim=0)

        # # 合并state和index_tensor为一个数据集
        # combined_dataset = TensorDataset(dataset_state, task_state_index)
        # # 使用DataLoader从合并后的数据集中采样
        # dataloader = DataLoader(combined_dataset, batch_size=200, shuffle=True)
        # # 遍历dataloader获取每次的采样
        # for batch_data in dataloader:
        #     sampled_state, sampled_index = batch_data
        #     # 在这里进行你需要的操作
        #     print(sampled_state)
        #     print(sampled_index)

        return dataset_state,task_state_index

    def gen_total_instances(self, num_task):
        # TODO 默认前num_task个task

        list_task = []
        for i in range(num_task):
            list_task.append(i)

        dataset_state = []
        task_state_index = []

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
        # [num_task,num_state,128] -> [num_state,num_task,128]
        y = torch.stack(y_list, dim=0).transpose(0, 1)

        # [num_task,num_state,2] -> [num_state,num_task,2]
        index = torch.stack(index_list, dim=0).transpose(0, 1)


        # 找到矩阵中的最小值和最大值
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
        return torch.exp(-x / decay_rate) * 10000


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
        reward = (1.0 * reward_delay) + (0.25 * reward_add_delay) + (0.25 * reward_sub_delay)

        reward = (reward - reward.mean()) / (reward.std() + 1e-9)

        # 无下一个状态
        state = None
        # 一次即结束
        done = None
        return state, reward, done

    def complete_delay(self,action,index):

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
        # 延迟越小越好，取负值
        list_total_delay


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



