import numpy as np
import math
import torch
import random

class RP:
    def __init__(self,d_r=500,d_m=3,avg_deg=3,sigma=0.15,rou=0.4):
        return None
    


class Reservoir:
    def __init__(self,rou=0.4,d_r=500,avg_degree = 3):
        self.d_r = d_r
        self.avg_degree = avg_degree
        self.adj,self.states = self.init_graph(rou,d_r,avg_degree)

    def init_graph(self,rou,d_r,avg_degree):
        states=np.random.rand(d_r)
        adj=[]
        for i in range(d_r):
            connection=np.zeros(d_r)
            degree = max(0,round(np.random.normal(avg_degree,avg_degree/3,1)[0]))
            #print(i,degree)
            connected_idx = random.sample(range(d_r-1),degree)
            #print(connected_idx)
            for idx in connected_idx:
                connection[idx]=random.uniform(-1,1)
            adj.append(connection)

        max_eig=np.abs(max(np.linalg.eig(adj)[0]))
        for i in range(d_r):
            for j in range(d_r):
                adj[i][j]*=rou/max_eig
        return adj,states
    
    def update(self,feed):
        #print(self.states)
        state_update = np.tanh(np.dot(self.adj,self.states))
        #print("res:",state_update)
        self.states += state_update
    
    def r_star(self):
        r_star=[]
        for i in range(len(self.states)):
            if i%2 == 0:
                r_star.append(self.states[i]*self.states[i])
            else:
                r_star.append(self.states[i])

    def states(self):
        return self.states
    
    def adj(self):
        return self.adj