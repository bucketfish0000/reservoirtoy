import input,reservoir,output
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

def training_routine(system:list,cutoff:int,inputlyr:input.input,res:reservoir.Reservoir,outputlyr:output.output,lr):
    print("training")
    predictions=[]
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(params=outputlyr.out.parameters(),lr=lr,weight_decay=0.002)
    for i in range(cutoff):
        system_state=system[i]
        system_state_nxt=torch.FloatTensor(system[i+1])
        feed_to_res = inputlyr.to_reservoir(system_state)
        res.update(feed_to_res)
        #prediction=outputlyr.train(res_feed=res.states,system_state_nxt=system_state_nxt,loss_fn=loss_fn,optimizer=optimizer)
        
        #moving training func to the loop instead of calling train()
        #commenting the following and uncomment line 21 makes call to train() which results in not 0-output but static output equal to the last output when training.
        ####
        prediction=torch.FloatTensor(outputlyr.out(torch.FloatTensor((feed_to_res))))
        loss = loss_fn(prediction,system_state_nxt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ####
        predictions.append(prediction.detach().numpy())
    return predictions

def predict_routine(prev_run:list,end:int,inputlyr:input.input,res:reservoir.Reservoir,outputlyr:output.output):
    print("running preds")
    predictions=[]
    last_pred=prev_run[-1]
    for i in range(end):
        feed_to_res = inputlyr.to_reservoir(last_pred)
        res.update(feed_to_res)
        #prediction=outputlyr.predict(res_feed=res.states)
        with torch.no_grad():
            prediction=torch.FloatTensor(outputlyr.out(torch.FloatTensor((feed_to_res))))
        predictions.append(prediction.detach().numpy())
        last_pred=prediction
    #print(prev_run[-2],prev_run[-1],predictions[0])
    return predictions

def plot(values,delta_t,dimension=3):
    t = 0
    time=[]
    f = plt.figure()
    f.set_figwidth(40)
    f.set_figheight(20)
    for i in range(len(values[0])):
        time.append(t)
        t+=1
    
    for seq in values:
        data = []
        for i in range(dimension):
            data.append([])   
        for vect in seq:
            for i in range(dimension):
                data[i].append(vect[i])
        plt.subplot(311)
        plt.plot(time,data[0])
        plt.subplot(312)
        plt.plot(time,data[1])
        plt.subplot(313)
        plt.plot(time,data[2])
    
    plt.show()

def main(d_r,tp,lr):

    inputlyr = input.input(out_dim=d_r)
    res=reservoir.Reservoir(d_r=d_r)
    outputlyr=output.output(res_feed_dim=d_r)

    lor = input.lorenz()
    #plot(lor,0.01,3) 

    cutoff = (int)(round(len(lor)*tp))
    end = (int)(round(len(lor)-cutoff))

    training_output=training_routine(lor,cutoff,inputlyr,res,outputlyr,lr=lr)
    prediction = predict_routine(training_output,end,inputlyr,res,outputlyr)
    #print(prediction)
    whole = training_output+prediction
    print("doing plots")
    plot([lor,whole],0.01,3)
    return inputlyr,res,outputlyr,training_output,prediction,lor