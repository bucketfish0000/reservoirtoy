import numpy as np
import math
import torch
import random

class Reservoir:
    def __init__(self,rou=0.4,d_r=500,avg_degree = 3):
        self.d_r = d_r
        self.avg_degree = avg_degree
        self.adj,self.states = self.init_graph(rou,d_r,avg_degree)



    def _init_graph(rou,d_r,avg_degree):
        states=np.zeros(d_r)
        adj=[]
        for i in range(d_r):
            connection=np.zeros(d_r)
            degree = round(max(0,round(random.normal(avg_degree,avg_degree/3,1))))
            connected_idx = random.sample(range(d_r-1),degree)
            for idx in connected_idx:
                connection[idx]=random.randrange(-1,1)
            adj.append(connection)
        max_eig=max(np.linalg.eig(adj)[0])
        for i in range(avg_degree):
            for j in range(avg_degree):
                adj[i][j]*=rou/max_eig
        return adj,states
    
    def update(self,feed):
        self.states += np.tanh(np.dot(self.adj,self.states))
    
