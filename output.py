import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class outputLayer(nn.Module):
    def __init__(self,res_feed_dim=500,output_dim=3):
        super().__init__()
        self.lin=nn.Sequential(
            nn.Linear(res_feed_dim,output_dim)
        )
    
    def forward(self,x):
        x=self.lin(x)
        return x

class output:
    def __init__(self,res_feed_dim=500,output_dim=3):
        self.out = outputLayer(res_feed_dim,output_dim)
        self.wout = np.random.rand(output_dim,res_feed_dim)

    def train(self,res_feed,system_state_nxt,loss_fn,optimizer):
        prediction=self.out(torch.FloatTensor((res_feed)))
        loss = loss_fn(torch.FloatTensor(prediction),torch.FloatTensor(system_state_nxt))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return prediction
    def predict(self,res_feed):
        with torch.no_grad():
            prediction=self.out(torch.FloatTensor((res_feed)))
        return prediction
    
    
