import torch 
import torchvision
import torch.nn as nn 
import os
import numpy as np 

def calculate_mae(predictions, targets):
    mae = torch.mean(torch.abs(predictions - targets))
    return mae


def train_model(dataloader, model, loss_fn,metric,  optimizer,epochs):
    history = dict()
    loss_epochs  = []
    mae_epochs = []
    best_mae = -torch.inf
    for epoch in range(epochs):
        num_batches = len(dataloader)
        total_loss = 0
        total_mae = 0 
        model.train()
        
        for x_batch, y_batch in dataloader:
            predict = model(x_batch.to(torch.float32))
            loss  = loss_fn(predict, y_batch.to(torch.float32))
            mae = metric(predict, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            
            total_loss += loss.item()
            total_mae += mae  
            # if mae > best_mae:
            #     best_mae = mae
            #     best_weight_path =  "./work_dir/model_weight/" + "best_weight"
            #     os.makedirs(best_weight_path, exist_ok=True)
            #     torch.save(model.state_dict(), best_weight_path)
        loss_e  = total_loss / num_batches
        mae_e = total_mae / num_batches
        
        print(f"Epoch: {epoch} | Loss: {loss_e} | MAE: {mae_e}")
        loss_epochs.append(loss_e)
        mae_epochs.append(mae_e)
    history['loss'] = loss_epochs
    history['MAE'] = mae_epochs
    
    return history, model