import random
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

class Data:

    def __init__(self,num_task=5):

        # 任务数
        self.num_task = num_task
        # 每个任务的状态数  workload pattern
        self.num_state = 3
        self.list_state = list(range(self.num_state))

        # CPU
        self.max_num_cpu = 50
        # 缓存
        self.max_cache = 128.
        # I/O 百分比
        self.max_io = 100.

        # # 一个任务下每个状态batch_size，共batch_size*3
        # state_data, y = self.gen_train_dataset2(batch_size=1024)

    def compute(self,x,k,decay_rate):

        batch_size = x.size(0)
        cpu_io_cache = x[:,1:]
        state = x[:,0]
        # cpu = x[:,1]
        # io = x[:,2]
        # cache = x[:,3]


        # state1
        index = torch.nonzero(state == 0.).squeeze()
        y1 = k[0] * torch.exp(-cpu_io_cache[index] / decay_rate[0])
        y1 = torch.sum(y1,dim=-1)

        # state2
        index = torch.nonzero(state == 1.).squeeze()
        y2 = k[1] * torch.exp(-cpu_io_cache[index] / decay_rate[1])
        y2 = torch.sum(y2,dim=-1)

        # state3
        index = torch.nonzero(state == 2.).squeeze()
        y3 = k[2] * torch.exp(-cpu_io_cache[index] / decay_rate[2])
        y3 = torch.sum(y3,dim=-1)
        # 生成的x中state是按顺序排序的
        y =torch.cat((y1,y2,y3))

        return y



    def gen_train_dataset(self,batch_size):


        decay_rate = []
        k = []
        for state in range(self.num_state):
            temp_decay_rate = []
            temp_k =[]
            for x in range(3):
                # 随机生成一个下降速率 [0,1)
                decay_rate_ = random.random() * 500
                temp_decay_rate.append(decay_rate_)
                k_ = random.random() + 0.5
                temp_k.append(k_)
            decay_rate.append(temp_decay_rate)
            k.append(temp_k)

        decay_rate = torch.tensor(decay_rate)
        k = torch.tensor(k)


        # [0,51)
        cpu = torch.randint(0,self.max_num_cpu+1,(batch_size,))
        # [0,1) * 100.
        io = torch.rand(batch_size) * self.max_io
        # [0,1) * 128.
        cache = torch.rand(batch_size) * self.max_cache

        cpu= cpu.unsqueeze(-1)
        io = io.unsqueeze(-1)
        cache = cache.unsqueeze(-1)

        data = torch.cat((cpu,io,cache),dim=-1)

        # import matplotlib.pyplot as plt
        # # 提取张量中的数据
        # data = data.numpy()
        # # 创建一个三维图形
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # # 绘制数据点
        # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o')
        #
        # # 设置图形属性
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.set_title('3D Scatter Plot')
        # plt.show()

        # data = data.unsqueeze(0).repeat(3,1,1)
        state = torch.tensor([0,1,2]).unsqueeze(-1).repeat(1,1,batch_size).squeeze()

        # [batch_size, 4]
        s1 = torch.cat((state[0].unsqueeze(-1),data),dim=-1)
        s2 = torch.cat((state[1].unsqueeze(-1),data),dim=-1)
        s3 = torch.cat((state[2].unsqueeze(-1),data),dim=-1)


        # [3*batch_size, 4]
        state_data = torch.cat((s1,s2,s3),dim=0)

        y = self.compute(state_data,k,decay_rate)

        return state_data,y


    def gen_train_dataset2(self,batch_size):


        decay_rate = []
        k = []
        for state in range(self.num_state):
            temp_decay_rate = []
            temp_k =[]
            for x in range(3):
                # 随机生成一个下降速率 [0,1)
                decay_rate_ = random.random() * 500
                temp_decay_rate.append(decay_rate_)
                k_ = random.random() + 0.5
                temp_k.append(k_)
            decay_rate.append(temp_decay_rate)
            k.append(temp_k)

        decay_rate = torch.tensor(decay_rate)
        k = torch.tensor(k)


        # [0,50]
        cpu =  torch.linspace(0, 1, self.max_num_cpu) *  self.max_num_cpu
        #  100.
        io = torch.linspace(0, 1, int(self.max_io)) *  self.max_io
        #  128.
        cache = torch.linspace(0, 1, int(self.max_cache)) *  self.max_cache

        # 将张量扩展为合适的形状
        cpu = cpu.view(self.max_num_cpu, 1, 1)
        io = io.view(1, int(self.max_io), 1)
        cache = cache.view(1, 1, int(self.max_cache))

        # 连接三个张量
        combined_tensor = torch.cat((cpu, io, cache), dim=1)

        # 重新塑造为[50*128*100,3]的形状
        combined_tensor = combined_tensor.view(50 * 128 * 100, 3)


        cpu= cpu.unsqueeze(-1)
        io = io.unsqueeze(-1)
        cache = cache.unsqueeze(-1)

        data = torch.cat((cpu,io,cache),dim=-1)

        # data = data.unsqueeze(0).repeat(3,1,1)
        state = torch.tensor([0,1,2]).unsqueeze(-1).repeat(1,1,batch_size).squeeze()

        # [batch_size, 4]
        s1 = torch.cat((state[0].unsqueeze(-1),data),dim=-1)
        s2 = torch.cat((state[1].unsqueeze(-1),data),dim=-1)
        s3 = torch.cat((state[2].unsqueeze(-1),data),dim=-1)

        # [3*batch_size, 4]
        state_data = torch.cat((s1,s2,s3),dim=0)

        y = self.compute(state_data,k,decay_rate)

        return state_data,y


if __name__ == '__main__':
    da = Data()
    da.gen_train_dataset(1024*10)










