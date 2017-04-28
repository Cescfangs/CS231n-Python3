import numpy as np


class BP_Network():
    def __init__(self, input_size, hidden_size, output_size, std=0.01, b_init=0):
        '''
        一输入 -> ReLu -> Softmax -> 输出
        '''
        self.params = {}
        # 从标准高斯分布中（期望为0，方差为1）随机取样作为 w 的初始值
        # 关于 np.random.randn，参见 https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html
        self.params['w_1'] = std * np.random.randn(input_size, hidden_size)
        self.params['w_2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b_1'] = np.full((hidden_size, 1), b_init)
        self.params['b_2'] = np.full((output_size, 1), b_init)

    def loss_and_gradient(X, target, reg=0.01):
        '''
        在这里要完成向前传播的过程，目的是计算 loss 和 gradient
        X: 训练的数据，(NxD)
        target: 训练数据的类型，(Nx1)
        '''
        w_1 = self.params['w_1']
        w_2 = self.params['w_2']
        b_1 = self.params['b_1']
        b_2 = self.params['b_2']
        # 第一层，输入 -> 隐含层
        out_1 = X.dot(w_1) + b_1
        a_1 = np.maximum(out_1, 0)

        out_2 = a_1.dot(w_2) + b_2
        a_2 = a_2 - np.max(a_2, axis=1)
        
