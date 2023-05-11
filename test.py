import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from modelSwell import Model
import pandas as pd
from torch.utils.data import TensorDataset,DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import recall_score, f1_score, classification_report, confusion_matrix
np.random.seed(12)
BATCH_SIZE=16
epochs = 2000
save_path = 'Model_f40'



def get_avg(list):
    sum = 0
    for l in range(0, len(list)):
        sum = sum + list[l]
    return sum / len(list)

def load_testdata():
    
    path = './data/SwellHLANorthS5_part.mat' #
    data = sio.loadmat(path)
    data = np.array(data['SwellHLANorthS5']).astype(np.float64)
    data = data[2401:3000,:]
    LengthData=len(data)

    test_data=data[:int(LengthData),0:-1]
    test_lable=data[:int(LengthData), -1].astype(np.int64)
    # test_lable=test_lable-17
    
    a = torch.tensor(test_data)
    b = torch.tensor(test_lable)
    testload = TensorDataset(a, b)
    test_loader = DataLoader(dataset=testload, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader
    
def draw_fig(list,name,epoch):
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=15)
        plt.plot(x1, y1, '-')
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('Train loss', fontsize=15)
        plt.show()
    elif name =="acc":
        plt.cla()
        plt.title('Val accuracy vs. epoch', fontsize=15)
        plt.plot(x1, y1, '-')
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('Val accuracy', fontsize=15)
        plt.show()


    
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader=load_testdata()


# -------------------------------------------Load---------------------------------------------------------
    # Model class must be defined somewhere
    # net = torch.load('Model')
    net = torch.load(save_path)
    # net.eval()


    for p in net.parameters(): 
        p.requires_grad = False # Set to False means that only the weights of the final fully connected layer are trained, and the rest of the layers are not trained
    num_classes = 6 
    inchannel = net.classifier[4].out_features
    net.classifier[7] = nn.Linear(inchannel, num_classes) 


    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.00001)
    trainloss=[]
    
    best_acc = 0.0
    val_trues=[]
    val_preds = []
    testacc=[]
    test_steps = len(test_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        tacc=0.0
        train_bar = tqdm(test_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            images=images.view(-1,91,36)
            #print(images.shape)
            # images=images.unsqueeze(1).float()
            images=images.float()
            optimizer.zero_grad()
            outputs = net(images.to(device))

            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            y_outputs = net(images.to(device))
            y = torch.max(y_outputs, dim=1)[1]
            tacc += torch.eq(y, labels.to(device)).sum().item()
            train_acc = tacc / (len(test_loader)*BATCH_SIZE)
            testacc.append(train_acc)
            
            val_outputs = outputs.argmax(dim=1)
            val_preds.extend(val_outputs.detach().cpu().numpy())
            val_trues.extend(labels.detach().cpu().numpy())
            sklearn_recall = recall_score(val_trues, val_preds, average='micro')
            sklearn_f1 = f1_score(val_trues, val_preds, average='micro')

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}\t train_acc={:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss,train_acc)



        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / test_steps, train_acc))
        trainloss.append(running_loss / test_steps)
        
            


if __name__ == '__main__':
    my_filename = os.path.basename(__file__)
    main()
