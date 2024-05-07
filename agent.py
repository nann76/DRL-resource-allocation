import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model import Model
from mlp import MLP


class ActorCritic(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.num_layers_actor =3     # MLP 的层数
        self.num_layers_critic = 3

        self.hidden_dim_actor = 256        # 隐藏层的维度
        self.hidden_dim_critic = 256

        self.input_dim_actor = 128     # 输入维度
        self.input_dim_critic = 128

        self.output_dim_actor = 1     # 输出维度
        self.output_dim_critic = 1

        self.actor = MLP(num_layers=self.num_layers_actor,
                         input_dim=self.input_dim_actor,
                         hidden_dim=self.hidden_dim_actor,
                         output_dim=self.output_dim_actor)

        self.critic = MLP(num_layers=self.num_layers_critic,
                          input_dim=self.input_dim_critic,
                          hidden_dim=self.hidden_dim_critic,
                          output_dim=self.output_dim_critic)



class A2C_Agent:
    def __init__(self):


        # self.device = 'cpu'

        self.gamma = 1.0      # 折扣率

        self.lr= 2e-4


        self.model = ActorCritic()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.Mseloss = nn.MSELoss()



    def get_action(self, state):

        scores = self.model.actor(state).squeeze()
        action_probs = F.softmax(scores,dim=-1)

        # dist_action = Categorical(action_probs)

        # action_index = dist_action.sample()
        # log_prob = dist_action.log_prob(action_index)

        return action_probs

    def learn(self, state, reward, action_prob):
        # 使用Critic网络估计状态值
        # h_critic = state.mean(dim=-2)
        # state_value = self.model.critic(h_critic).squeeze()


        state_action_value = self.model.critic(state).squeeze()
        state_value = torch.sum(state_action_value * action_prob,dim=-1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-9)
        advantage = reward.detach() - state_value.detach()

        # 计算Actor网络的损失并进行更新
        log_action_probs = torch.log(action_prob)
        loss_actor = torch.mean( - log_action_probs  *  advantage.unsqueeze(-1)  ) # 使用资源分配比例和实际奖励计算损失

        # loss_actor = -torch.log(action_prob) * (reward - v.detach())  # 使用资源分配比例和实际奖励计算损失


        # 计算Critic网络的损失并进行更新
        critic_loss = self.Mseloss(state_value,reward)
        loss = loss_actor + 0.5* critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()





if __name__ == '__main__':

    from environment import Env
    from torch.utils.data import DataLoader, TensorDataset

    env = Env()
    state,index = env.train_dataset

    agent = A2C_Agent()

    # 合并state和index_tensor为一个数据集
    combined_dataset = TensorDataset(state, index)
    # 使用DataLoader从合并后的数据集中采样
    dataloader = DataLoader(combined_dataset, batch_size=200, shuffle=True)
    # 遍历dataloader获取每次的采样
    for batch_data in dataloader:
        sampled_state, sampled_index = batch_data


        action_probs = agent.get_action(sampled_state)
        _,reward,_ = env.step(action_probs,sampled_index)
        print(reward)
        # agent.learn(state,reward=torch.tensor(10.),action_prob=action_probs)



