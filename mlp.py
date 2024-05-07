import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.num_layers = num_layers

        self.linears = torch.nn.ModuleList()
        '''
        self.batch_norms = torch.nn.ModuleList()
        '''

        self.linears.append(nn.Linear(input_dim, hidden_dim))

        for layer in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))

        self.linears.append(nn.Linear(hidden_dim, output_dim))
        '''
        for layer in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
        '''
        # self.init_weight()
    # def init_weight(self):
    #     for layer in self.linears:
    #         torch.nn.init.normal_(layer.weight, mean=0, std=1)

    def forward(self, x):

        # If MLP
        # print(x.size())
        h = x
        for layer in range(self.num_layers - 1):
            '''
            h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            '''
            h = torch.relu(self.linears[layer](h))
            # print(h.size())
            # h = F.relu(self.linears[layer](h))
        return self.linears[self.num_layers - 1](h)

