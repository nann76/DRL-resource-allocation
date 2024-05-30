import copy
import os
import pandas as pd
import torch
import random
import time
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader, TensorDataset
from pre_model import Pre_MLP
from KAN import KAN
from gen_instances import Data
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




class TrainManager:

    def __init__(self):

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')

        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # torch.set_default_dtype(torch.float64)
        else :
            torch.set_default_tensor_type('torch.FloatTensor')
            num_cpus = torch.get_num_threads()
            print(f"PyTorch默认使用的cPU核个数为：{num_cpus}")
        # print("Pytorch device: ",device.type)
        print('Using PyTorch version:', torch.__version__, ' Device:', self.device)




    def train(self,task=0):


        max_epoch =  200
        maxlen_best_model = 1  # Save the best model
        loss_best =float('inf')
        last_best_model_path = None
        count_iters = 0
        list_loss = []

        str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        save_dir = f'./train_dir/task_{task}_{str_time}'
        os.makedirs(save_dir)

        model = Pre_MLP(input_dim=3, hidden_dim=64, output_dim=1,dropout_prob=0.2).to(self.device)
        # model = KAN([4,64,1]).to(self.device)
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        lr = 1e-3
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                  step_size=configs.decay_step_size,
        #                                                  gamma=configs.decay_ratio)

        # 训练数据
        # gen_data = Data()
        #
        # state_data, y = gen_data.gen_train_dataset(batch_size=1024*10)
        # state_data = state_data[0:1024*10,1:]
        # y = y[0:1024*10]
        # 数据处理
        # gen_data = Data()
        #
        # state_data, y = gen_data.all()

        cpu = np.arange(1,51,2)  # 0 to 50
        io = np.arange(1,101,2)  # 0 to 100
        cache = np.arange(1,129,2)  # 0 to 128
        cpu_grid, io_grid, cache_grid = np.meshgrid(cpu, io, cache, indexing='ij')
        combinations = np.vstack([cpu_grid.ravel(), io_grid.ravel(), cache_grid.ravel()]).T

        def f(cpu_io_cache):
            return (np.exp(-cpu_io_cache[:, 0] / 24) +
                    np.exp(-cpu_io_cache[:, 1] / 31) +
                    np.exp(-cpu_io_cache[:, 2] / 5)) * 200

        X = combinations
        y = f(X)

        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        val_size = len(y_test)
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()


        # 合并state和index_tensor为一个数据集
        combined_dataset_train = TensorDataset(X_train, y_train)
        combined_dataset_test = TensorDataset(X_test, y_test)



        # 使用DataLoader从合并后的数据集中采样
        train_dataloader = DataLoader(combined_dataset_train, batch_size=1024, shuffle=True)
        val_dataloader =  DataLoader(combined_dataset_test, batch_size=val_size)

        start_train_time = time.time()

        for i in range(1,max_epoch + 1):
            print(f'epoch: {i}')



            # 遍历dataloader获取每次的采样
            for batch_data in train_dataloader:
                sampled_state_data, sampled_y = batch_data
                sampled_state_data = sampled_state_data.to(self.device)
                sampled_y = sampled_y.to(self.device)

                model.train()

                y = model(sampled_state_data)

                # 模型更新
                optimizer.zero_grad()
                loss = criterion(y, sampled_y)
                loss.backward()
                optimizer.step()
                print(f'iter: {count_iters} ,train loss : {loss}')

                count_iters +=1

                if count_iters % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        for batch_data in val_dataloader:
                            sampled_state_data, sampled_y = batch_data
                            sampled_state_data = sampled_state_data.to(self.device)
                            sampled_y = sampled_y.to(self.device)
                            y = model(sampled_state_data)
                            loss = criterion(y, sampled_y)
                            list_loss.append(loss)
                            print(f'iter: {count_iters} ,val  loss : {loss}')

                            if loss < loss_best:

                                loss_best = loss
                                save_new_model_path = '{0}/model_T{1}_I{2}.pt'.format(save_dir, task, count_iters)
                                if last_best_model_path != None:
                                    os.remove(last_best_model_path)
                                last_best_model_path = save_new_model_path

                                torch.save(model.state_dict(), save_new_model_path)

                    ExpLR.step()






        seconds = time.time() - start_train_time
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = int(seconds % 60)
        print(f"total_train_time: H:{hours}-M:{minutes}-S:{remaining_seconds}")
        print(list_loss)

        with open('{0}/list.txt'.format(save_dir),'w') as f:

            f.write('valid_list_mean_delay' + str(list_loss) + '\n\n')

        import matplotlib.pyplot as plt

        # plt.switch_backend('Agg')
        plt.figure(figsize=(12, 6))
        x_data = list(range(1,len(list_loss)+1))
        plt.plot( x_data, list_loss, label='mean delay')

        plt.xlabel('iterations')
        plt.ylabel('mean delay')
        plt.legend()

        plt.grid()  # 网格
        plt.tight_layout()  # 去白边
        plt.savefig(save_dir+'/mean_delay.png', dpi=200)
        plt.show()






if __name__ == '__main__':

    t = TrainManager()

    task = 0
    setup_seed(task * 55)

    t.train(task)
