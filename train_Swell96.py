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
from sklearn.metrics import recall_score, f1_score, classification_report, confusion_matrix
import copy
np.random.seed(12)
BATCH_SIZE=16
LengthData=3000
epochs = 20


def load_data():
    path = './data/SwellHLANorthS5_part.mat' #
    data = sio.loadmat(path)
    data = np.array(data['SwellHLANorthS5']).astype(np.float64)  
    # train
    trainData = data[0:LengthData,:] 
    index = [i for i in range(len(trainData))] 
    np.random.shuffle(index) # Disrupting the index
    trainData = trainData[index]

    train_data=trainData[:int(LengthData*0.8),0:-1]
    train_lable=trainData[:int(LengthData*0.8), -1].astype(np.int64)
    test_data = trainData[int(LengthData * 0.8):, 0:-1]
    test_lable = trainData[int(LengthData * 0.8):, -1].astype(np.int64)

    
    a = torch.tensor(train_data)
    b = torch.tensor(train_lable)
    train_load = TensorDataset(a, b)
    train_loader = DataLoader(dataset=train_load, batch_size=BATCH_SIZE, shuffle=True)
    c = torch.tensor(test_data)
    d = torch.tensor(test_lable)
    test_load = TensorDataset(c, d)
    test_loader = DataLoader(dataset=test_load, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader,test_loader
  
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

# # -----------Load the optimal weights to filter out the non-reusable parts of the local model---------------
def para_state_dict(net, model_save_path):
    state_dict = copy.deepcopy(net.state_dict()) # copy the original parameters in the network
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        for key in state_dict:  # Iterate over the corresponding parameters in the new network model and replace the original parts with those that can be reused in SparseAutoencoder
            # print(key)                
            if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():  
                print("Successful initialization of parameters:", key)
                state_dict[key] = loaded_paras[key]
    return state_dict

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader,val_loader=load_data()

    net = Model(num_classes=63, init_weights=True)
    model_save_path = 'SparseAutoencoderModel_SwellHLANorthS5_part.pth'
    state_dict = para_state_dict(net, model_save_path)
    net.load_state_dict(state_dict)
  

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.00001)
    trainloss=[]
    train_trues=[]
    train_preds = []
    valacc=[]
    save_path = './Model_part_afterSAE'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        tacc=0.0
        train_bar = tqdm(train_loader)
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
            
            train_outputs = outputs.argmax(dim=1)
            train_preds.extend(train_outputs.detach().cpu().numpy())
            train_trues.extend(labels.detach().cpu().numpy())
            
            sklearn_recall = recall_score(train_trues, train_preds, average='micro')
            sklearn_f1 = f1_score(train_trues, train_preds, average='micro')

            y_outputs = net(images.to(device))
            y = torch.max(y_outputs, dim=1)[1]
            tacc += torch.eq(y, labels.to(device)).sum().item()
            train_acc = tacc / (len(train_loader)*BATCH_SIZE)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}\t train_acc={:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss,train_acc)

        # validate
        net.eval()
        val_trues=[]
        val_preds = []
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                #print(val_images.shape)
                val_images = val_images.view(-1, 91, 36)
                val_images = val_images.float()
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                
                val_outputs = outputs.argmax(dim=1)
                val_preds.extend(val_outputs.detach().cpu().numpy())
                val_trues.extend(val_labels.detach().cpu().numpy())
                sklearn_recall = recall_score(val_trues, val_preds, average='micro')
                sklearn_f1 = f1_score(val_trues, val_preds, average='micro')


        val_accurate = acc / (len(val_loader)*BATCH_SIZE)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        trainloss.append(running_loss / train_steps)
        valacc.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
   
    print('Finished Training')

    

if __name__ == '__main__':
    my_filename = os.path.basename(__file__)
    main()
