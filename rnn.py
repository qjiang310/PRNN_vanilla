import numpy as np
from numpy import random

class RNN(object):
    def __init__(self, input_dim, hidden_dim, output_dim, timesteps, batch_size, iter, lr=0.002):
        self.lr = lr
        self.timesteps = timesteps 
        self.batch_size = batch_size
        self.iter = iter
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # self.U = np.random.uniform(0, 1, (self.hidden_dim, self.hidden_dim)) 
        # self.U = np.zeros((self.hidden_dim, self.hidden_dim)) + 0.5
        self.U = self.loadWeight()
        # print(np.array(self.U))

        self.h_prev = self.loadActivation(True)
        # print(np.array(self.h_prev))

        self.Wx = self.loadActivation(False)
        # print(np.array(self.Wx))

        # tmp variable
        self.H = None
        self.Alpha = None

    def forward_prop(self):
        # Wx = np.zeros((batch_size, self.hidden_dim)) # TODO receive from PRNN
        # Wx = np.random.uniform(0, 1, (self.timesteps, batch_size, self.hidden_dim))
        # self.Wx = np.zeros((self.iter, self.hidden_dim, self.batch_size)) + 0.5
        self.H = np.zeros((self.iter, self.hidden_dim, self.batch_size))
        self.Alpha = np.zeros((self.iter, self.hidden_dim, self.batch_size))
        # h_prev = np.zeros((self.timesteps, self.hidden_dim, batch_size)) + 0.5 # TODO receive from PRNN
        # h_prev = np.random.uniform(0, 1, (batch_size, self.hidden_dim))
        sigmoid_output = np.zeros((self.iter, self.batch_size, self.output_dim))
        for t in range(self.iter):
            self.Alpha[t] = np.dot(self.U, self.h_prev) + self.Wx[t]
            self.H[t] = self.relu(self.Alpha[t]) # be consistent with PRNN
            self.h_prev = self.H[t]
        return self.H

    def sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def relu(self, x):
        return np.maximum(x, 0)

    def loadWeight(self):
        f = open("weights.txt", "r")
        return np.loadtxt(f)
        
    def loadActivation(self, is_h_prev):
        f2 = open("inActivation.txt", "r")
        arr = np.loadtxt(f2).reshape(self.timesteps, self.hidden_dim, self.batch_size)
        if (is_h_prev):
            return arr[0,:,:]
        else:
            return arr[1:self.timesteps,:,:]

def main():
    timesteps = 4; # in PRNN iterations = batch_size * (timesteps - 1)
    # data_size = 1000
    # split_ratio = 0.9 # TODO ? 
    max_iter = 3 # iterations in PRNN
    batch_size = 3
    rnn = RNN(1, 4, 1, timesteps, batch_size, max_iter)
    
    for iters in range(timesteps): # might be timesteps - 1
        H = rnn.forward_prop()
        print(np.array(H))
    # for iters in range(max_iter+1):
    #     sigmoid_output = rnn.forward_prop(batch_size)
    #     predict_label = sigmoid_output > 0.5
    #     accuracy = float(np.sum(predict_label == 0)) / test_y.size
    #     print("The accuracy on testing data is %f" % accuracy)

    
        
if __name__ == '__main__':
    main()