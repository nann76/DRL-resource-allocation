import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy


from mlp import MLP




class Model(nn.Module):


    def __init__(self):
        super(Model, self).__init__()

        self.num_layers_actor =3     # MLP 的层数
        self.num_layers_critic = 3

        self.hidden_dim_actor = 128        # 隐藏层的维度
        self.hidden_dim_critic = 128



        self.input_dim_actor = 64     # 输入维度
        self.input_dim_critic = 64

        self.output_dim_actor = 1     # 输出维度
        self.output_dim_critic = 1


        self.device = 'cuda:0'




        self.actor = MLP(num_layers=self.num_layers_actor,
                         input_dim=self.input_dim_actor,
                         hidden_dim=self.hidden_dim_actor,
                         output_dim=self.output_dim_actor).to(self.device)

        self.critic = MLP(num_layers=self.num_layers_critic,
                         input_dim=self.input_dim_critic,
                         hidden_dim=self.hidden_dim_critic,
                         output_dim=self.output_dim_critic).to(self.device)



    def forward(self):
        pass





    # def get_action_prob(self,state,memory,flag_sample=True,flag_train=True):
    #
    #
    #
    #     h_action = state
    #     scores =self.actor(h_action).flatten(1)
    #     action_probs = F.softmax(scores,dim=1)





    def choose_action(self,state,memory,flag_sample=True,flag_train=True):
        '''
        :param state:
        :param memory:
        :param flag_sample:
        :param flag_train:
        :return:
        '''


        # action_probs, ope_step = self.get_action_prob(state, memory, flag_train=flag_train)

        scores = self.actor(state).flatten(1)
        action_probs = F.softmax(scores, dim=1)

        if flag_sample:
            # 根据动作概率分布随机采样
            dist_action = Categorical(action_probs)
            action_index = dist_action.sample()

        # greedy
        else:
            action_index = action_probs.argmax(dim=1)


        # Store data in replay -buffer during training
        if flag_train == True:
            memory.states.append(copy.deepcopy(state))
            memory.logprobs.append(dist_action.log_prob(action_index))
            memory.actions.append(action_index)

        return


    def evaluate(self,states_dict,actions):



        return








if __name__ == '__main__':
    pass


