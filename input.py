import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def lorenz(params=[10,28,8/3],init=[25,25,25], epoch=300,delta_t=0.01,dimension=3):
    '''
    generates lorenz system states of a certain period
    input:
        params - list of params used in func
        init - initial state of vars
        epoch - # of time slots to iterate
        delta_t - duration of each slot
        dimension - number of dims of vars 
    output:f = plt.figure()
f.set_figwidth(4)
f.set_figheight(1)
        result - list of tensors
    '''
    result = []
    result.append(init)
    '''
    discrete apprxm. for small dt:
    x(t+dt)=x(t)+dt*(a*(y(t)-x(t))f = plt.figure()
f.set_figwidth(4)
f.set_figheight(1))
    y(t+dt)=y(t)+dt*(x(t)*(b-z(t))-y(t))
    z(t+dt)=z(t)+dt*(x(t)*y(t)-c*z(t))
    '''
    for i in range(0,epoch):
        prev = result[i-1]
        curr = [prev[0]+delta_t*(params[0]*(prev[1]-prev[0])),
                prev[1]+delta_t*(prev[0]*(params[1]-prev[2])-prev[1]),
                prev[2]+delta_t*(prev[0]*prev[1]-params[2]*prev[2])]
        result.append(curr)
        #print(curr)
    return result

class input:
    def __init__(self,sigma=0.15,in_dim=3,out_dim=500):
        self.sigma,in_dim,out_dim = sigma,in_dim,out_dim
        self.w_in = self.init_input_weight(sigma,in_dim,out_dim)
        
    def to_reservoir(self,input):
        #print("inputs:",input,np.dot(self.w_in,input))
        return np.dot(self.w_in,input)

    def init_input_weight(self,sigma=0.15,in_dimension=3,out_dimension=500):
        w_in = []
        for i in range(out_dimension):
            weight = np.zeros(in_dimension)
            weight[random.randint(0,2)]=random.uniform(-sigma,sigma)
            w_in.append(weight)
        print(len(w_in),len(w_in[0]))
        return w_in