







### 问题描述

每个任务有5个状态，某一时刻只处于一个状态。随缓存分配的增加，延迟减小。

N个任务分配128G缓存，目标使得N个task总延迟最小。



### DRL

* 单步，可视为无动作选择。

* 使用在[0,128]均匀采样128个点构成的向量表示某一任务的某一状态的曲线。

* 构建规模无关的策略网络，输入n X 128，输出n X 1，经过softmax，动作选择概率为每个任务分配缓存的比例。



#### 状态

曲线在[0,128]上均匀采样128个点构成的向量。

#### 策略
策略网络是规模无关的，为128到1的映射。即，输入为 n X 128,输出为n X 1。经过softmax，每个任务的概率大小为分配缓存的比例。

#### 算法

REINFORCE 算法，增大奖励大动作概率，即任务分配比例。

```python
reward = reward.detach()
log_action_probs = torch.log(action_prob)
# 增大奖励大动作概率，即任务分配比例
loss_actor = - torch.sum( log_action_probs * reward,dim=-1).sum()
```

#### 奖励

```python
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
```
