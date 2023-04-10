import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def lorenz(params=[10,28,8/3],init=[25,25,25], epoch=30000,delta_t=0.01,dimension=3):
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
    return result

def plot(values,delta_t,dimension=3):
    t = 0
    time = []
    data = []
    for i in range(dimension):
        data.append([])
    #print(data)
    for vect in values:
        time.append(t)
        t+= delta_t
        d = vect
        for i in range(dimension):
            #print(i,d)
            data[i].append(d[i])
    f = plt.figure()
    f.set_figwidth(40)
    f.set_figheight(10)
    plt.subplot(311)
    plt.plot(time,data[0])
    plt.subplot(312)
    plt.plot(time,data[1])
    plt.subplot(313)
    plt.plot(time,data[2])
    plt.show()

def to_reservoir(w_in,input, in_dimension = 3, out_dimension = 500):
    return np.dot(w_in,input)

def init_input_weight(sigma=0.15,in_dimension=3,out_dimension=500):
    w_in = []
    for i in range(out_dimension):
        weight = np.zeros(in_dimension)
        weight[random.randint(0,2)]=random.uniform(-sigma,sigma)
        w_in.append(weight)
    return w_in