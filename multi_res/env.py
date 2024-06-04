import torch
import numpy as np
def decimal_to_binary_vector(decimal, bit_length=4):
    """
    将十进制数转换为定长二进制字符串，并将其转换为特征向量，长度为bit_length
    """
    bin_str = format(decimal, f'0{bit_length}b')  # 将十进制数转换为二进制字符串
    binary_vector = [int(bit) for bit in bin_str]  # 将二进制字符串转换为特征向量
    return np.array(binary_vector)

def one_hot_vector(idx,length=3):
    '''
    创建长度为length，索引idx处为1的one-hot code
    '''
    one_hot = np.zeros((length,))
    one_hot[idx] = 1
    return one_hot


class Env:


    def __init__(self):
        pass

        self.num_task = 5
        self.num_state = 3


        # (state_idx,key_value)
        self.t_s = [[(0, 0), (1, 5), (2, 10)],
                      [(0, 11), (1, 5), (2, 10)],
                      [(0, 0), (1, 5), (2, 10)],
                      [(0, 0), (1, 5), (2, 10)],
                      [(0, 0), (1, 5), (2, 10)],]


        self.state  = torch.zeros((5,3,3+4))

        for task in range(self.num_task):
            for state in range(self.num_state):
                state_idx,key_value  = self.t_s[task][state]

                vector = np.concatenate((one_hot_vector(idx=state_idx,length=3),
                                         decimal_to_binary_vector(decimal=key_value, bit_length=4))
                                        , axis=0)

                self.state[task,state] = torch.tensor(vector)

        print(1)




    def step(self,action):
        pass

if __name__ == '__main__':

    env =Env()