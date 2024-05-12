import torch.nn as nn


class Pre_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout_prob):
        super(Pre_MLP, self).__init__()

        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

        self.activate_func = nn.Sigmoid()


    def forward(self, x):
        x =  self.activate_func(self.dropout1(self.linear1(x)))
        x =  self.activate_func(self.dropout2(self.linear2(x)))
        out = self.linear3(x)
        return out