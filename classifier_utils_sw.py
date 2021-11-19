#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
# import pandas as pd
from copy import deepcopy
import time
import csv
from utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

metric = ['acc', 'spe', 'sensi', 'pre', 'npv', 'f1']

def eval(y_true, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    try: acc = (tp + tn) / (tp + tn + fp + fn)    
    except: acc = 0
    try: spe = tn / (tn + fp) # specificity
    except: spe = 0
    try: sensi = tp / (tp + fn) # sensitivity, recall    
    except: sensi = 0
    try: pre = tp / (fp + tp) # precision
    except: pre = 0
    try: npv = tn / (tn + fn)    
    except: npv=0
    try: f1 = 2 * (pre * sensi) / (pre + sensi)    
    except: f1 = 0
 
    return [acc, spe, sensi, pre, npv, f1]
        

def train(model,n_cls, n_epochs, trainloader,valloader, criterion, optimizer, scheduler,tri ,device):
    
    best_model = deepcopy(model)
    best_avg = -np.inf
    best_f1 = -np.inf
    best_epoch = 0 
    for epoch in range(n_epochs):
        model.train()
        start_perf_counter = time.perf_counter()
        start_process_time = time.process_time()
        print(tri)
        #print(f'n_epoch:{epoch}, lr:{scheduler.get_last_lr()}')
        print(f'n_epoch:{epoch}')

        running_loss = 0.0
        train_loss = 0.0
        running_acc = 0.0
        running_f1 = 0.0
        
        tr_y = []
        tr_pred = []
        for i,data in enumerate(trainloader, 0):

            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)

            if n_cls == 1:
                labels = labels.unsqueeze(1).long().to(device)
            else:
                labels = labels.long().to(device)
            
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if n_cls == 1:
                proba = torch.sigmoid(outputs)
            else:
                proba = torch.nn.Softmax(dim=1)(outputs)
                proba = np.argmax(proba.detach().cpu().numpy(),1)
            

            running_loss += loss.item()
            tr_y.append(labels.detach().cpu().numpy())
            tr_pred.append(proba)
            running_acc += accuracy_score(labels.detach().cpu().numpy(), proba)
            running_f1 += f1_score(labels.detach().cpu().numpy(),proba)
            
            train_loss += loss.item()
            
            if i%100 == 99:
                print(f'[epoch_{epoch+1}, batch_{i+1}] loss: {running_loss/100}, acc: {running_acc/100}, f1: {running_f1/100}')
                running_loss = 0.0
                running_acc = 0.0
                running_f1 = 0.0

        end_perf_counter = time.perf_counter()-start_perf_counter
        end_process_time = time.process_time()-start_process_time
        
    
        print(f'perf_counter : {end_perf_counter}')
        print(f'process_time : {end_process_time}')
        
       
        tr_y = np.concatenate(tr_y, 0)
        tr_pred = np.concatenate(tr_pred, 0)
        metric_list = eval(tr_y, tr_pred)
        avg_metric = sum(metric_list) / len(metric_list)
        print('train_loss:', train_loss/len(trainloader), 'train_acc', metric_list[0],'train_f1', metric_list[5], 'train_avg', avg_metric)

        valid_loss, valid_acc, valid_f1, valid_avg = test(model, n_cls,valloader, criterion, device)
        
        print('valid_loss:',valid_loss, 'valid_acc:', valid_acc, 'valid_f1:', valid_f1, 'valid_avg:', valid_avg )
        
        #torch.save(model.state_dict(), './model/'+tri+'_epoch'+str(epoch)+'.pth')
        #scheduler.step(valid_loss)
        scheduler.step()
        print('lr : ',optimizer.param_groups[0]['lr'])
        print('current epoch:', epoch)
        if valid_avg > best_avg:
            print(epoch, 'save weight')
            best_epoch = epoch
            best_avg = valid_avg
            best_model = deepcopy(model)
            torch.save(model.state_dict(), './check_point/'+tri+'/'+str(epoch)+'.pth')
    
    #torch.save(model.state_dict(), 'infer.pth')    

    return best_model, best_epoch


def test(model, n_cls, data_loader, criterion, device):
    model.eval()
    total_loss=0.0
    
    va_y = []
    va_pred = []
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            
            if n_cls==1:
                labels = labels.unsqueeze(1).long().to(device)
            else:
                labels = labels.long().to(device)
                

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            if n_cls == 1:
                proba = torch.sigmoid(outputs)
            else:
                proba = torch.nn.Softmax(dim=1)(outputs)
                proba = np.argmax(proba.detach().cpu().numpy(),1)
            
            total_loss += loss.item()
            va_y.append(labels.detach().cpu().numpy())
            va_pred.append(proba)
            
    va_y = np.concatenate(va_y, 0)
    va_pred = np.concatenate(va_pred, 0)
    metric_list = eval(va_y, va_pred)
    avg_metric = sum(metric_list) / len(metric_list)
           
    return total_loss/len(data_loader), metric_list[0], metric_list[5], avg_metric
    
# def submit(model,n_cls, file_name, data_loader, device):
#     model.eval()
    
#     results_df = pd.DataFrame()
#     with torch.no_grad():
#         for i,data in enumerate(data_loader, 0):
#             inputs, image_name = data['image'], data['image_name']

#             inputs = inputs.to(device)

#             outputs = model.forward(inputs)
            

#             if n_cls==1:
#                 proba = torch.sigmoid(outputs)
#                 #proba = np.where(proba.cpu() >= 0.5, 1, 0)
#             else:
#                 proba = torch.nn.Softmax(dim=1)(outputs)
#                 #proba = np.argmax(np.array(proba.cpu()),1)


#             #proba = torch.sigmoid(outputs)
#             #proba = torch.nn.Softmax(dim=1)(outputs)
#             #proba = np.where(proba.cpu() >= 0.5, 1, 0)
#             #proba = np.argmax(np.array(proba.cpu()),1)
            
            
#             for idx,_ in enumerate(inputs):
                
#                 if n_cls==1:
#                     row = [image_name[idx], proba[idx][0].cpu()]
#                 else:
#                     row = [image_name[idx], proba[idx][1].cpu().item()]
            
#                 row_df = pd.DataFrame([row], columns = ['filename', 'pred'])
#                 results_df = pd.concat([results_df, row_df])
            
#             if i%100 == 99:
#                 print(i)
                

#     results_df.to_csv('./result/'+file_name+'.csv', header=True, index=False)
    
        
def submit_probs(model, data_loader, device):
    model.eval()

    probs = []
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs= data['image']
            inputs = inputs.to(device)

            outputs = model.forward(inputs)

            prob = torch.nn.Softmax(dim=1)(outputs)
            probs.append(prob.cpu().numpy())
            
            
            if i%100 == 99:
                print(i)
        
        probs = np.concatenate(probs, 0)
    print(probs.shape)
    dic = {'probs' : probs[:, 1]}

    return dic


def out_label(model, data_loader, device):
    model.eval()

    label = []
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            label.extend(data['label'].cpu().numpy())
            
    return label

