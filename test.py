import torch
import random
import numpy as np
from environment import  Env
from agent import Agent
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

if device.type == 'cuda':
    torch.cuda.set_device(device)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else :
    torch.set_default_tensor_type('torch.FloatTensor')
    num_cpus = torch.get_num_threads()
    print(f"PyTorch默认使用的CPU核个数为：{num_cpus}")
print('Using PyTorch version:', torch.__version__, ' Device:', device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(200)



num_tasks = 20
env = Env(num_tasks=num_tasks)


# 加载模型
model_path = './train_dir/num_task_6_20240507_162525/model_T6_I163.pt'
model_path = './train_dir/num_task_6_20240508_204647/model_T6_I489.pt'
model_path = './train_dir/num_task_10_20240508_212618/model_T10_I637.pt'
state_dict = torch.load(model_path)
agent = Agent()






# 加载数据
state, index = env.validate_dataset

action_probs = agent.get_action(state)
delay = env.complete_delay(action_probs, index)

# print('delay: ', delay)
print('mean delay: ', torch.mean(delay,dim=-1))

total_mean_delay = torch.mean(delay).item()
print('validate mean delay: ', total_mean_delay)


# 加载训练模型后，对比
agent.model.load_state_dict(state_dict)

action_probs = agent.get_action(state)
delay = env.complete_delay(action_probs, index)

# print('delay: ', delay)
print('mean delay: ', torch.mean(delay,dim=-1))

total_mean_delay = torch.mean(delay).item()
print('validate mean delay: ', total_mean_delay)