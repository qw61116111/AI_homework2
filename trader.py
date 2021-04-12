import argparse
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv    
from torchvision import transforms
import random
import argparse
'''
raw_data="C://Users/Q56091087/Desktop/training.csv"
column_names = ['open','high','low','close']
data_csv = pd.read_csv(raw_data, names = column_names)

raw_test_data="C://Users/Q56091087/Desktop/testing.csv"

oooo="C://Users/Q56091087/Desktop/output.csv"
test_data_csv = pd.read_csv(raw_test_data, names = column_names)
'''
column_names = ['open','high','low','close']

#%%
num_day=14
num_pred_day=1

data_fold=['open','high','low','close']

label_fold=['open']
train_mean=[172.9643,174.1242,171.8836,173.0119]
train_std=[21.86505,21.94644,21.83843,21.89171]



class dataset(torch.utils.data.Dataset):

    def __init__(self,is_train=True):
        self.data=[]
        self.label=[]
        self.temp=[]

        if is_train:
            for i in range(len(data_csv[:num_train])):
                for j in range(len(data_fold)):
                    self.temp.append(data_csv[data_fold[j]][i])
                self.data.append(self.temp)
                self.temp=[]
            z=np.array(self.data).T
            for i in range(len(data_fold)):
                for j in range(len(data_csv[:num_train])):
                    z[i][j]-=train_mean[i]
                    z[i][j]/=train_std[i]
            self.data=[]
            self.data=z.T

            for i in range(len(data_csv[:num_train])):
                for j in range(len(label_fold)):
                    self.temp.append(data_csv[label_fold[j]][i])
                self.label.append(self.temp)
                self.temp=[]
            self.label=np.array(self.label)

    def __len__(self):
        return len(self.data)-num_day-num_pred_day+1
    
    def __getitem__(self, index):

        a=self.data[index:index+num_day]
        b=self.label[index+num_day:index+num_day+num_pred_day]

        return a,np.squeeze(b)


#%%
def test_in():
    data=[]
    label=[]
    temp=[]
    for i in range(len(data_csv)):
        for j in range(len(data_fold)):
            if i>=(len(data_csv)-num_day):
                temp.append(data_csv[data_fold[j]][i])
        if i>=(len(data_csv)-num_day):
            data.append(temp)
        temp=[]
    z=np.array(data).T
    for i in range(len(data_fold)):
        for j in range(len(data_csv[num_train-num_day:num_train])):
            z[i][j]-=train_mean[i]
            z[i][j]/=train_std[i]
    data=[]
    data=z.T
    data=data[np.newaxis,:]
    
    return torch.from_numpy(data)

def test_nor(test_data_csv):
    temp=[]
    test_data=[]
    test_z=[]
    for i in range(len(test_data_csv)):
        temp=[]
        for j in range(len(data_fold)):
            temp.append(test_data_csv[data_fold[j]][i])
        test_z.append(temp)
        
    zz=np.array(test_z).T
    
    for i in range(len(data_fold)):
            for j in range(len(test_data_csv)):
                zz[i][j]-=train_mean[i]
                zz[i][j]/=train_std[i]
    test_data=zz.T
    return test_data,test_z
#%%
epochs = 1200

batch_size = 64
val_size=200
hidden_size = 32
num_layers = 2
num_feature=len(data_fold)

class LSTM(nn.Module):
    def __init__(self,num_feature,hidden_size, num_layers):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=num_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
            )
    
        self.fc=nn.Linear(hidden_size,num_pred_day)

    def forward(self,inputs):
        out,(h_n,c_n)=self.lstm(inputs, None)
        outputs=self.fc(h_n[1])
        return  outputs


def MSE(y_pred,y_true): 
    return  torch.mean(torch.mean(((y_pred-y_true))**2))
#%%
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    data_csv=pd.read_csv(args.training, names = column_names)
    test_data_csv=pd.read_csv(args.testing, names = column_names)
    
    num_train=len(data_csv)
    test_data,test_z=test_nor(test_data_csv)
    #%%
    net=LSTM(num_feature ,hidden_size,num_layers)
    trainloader=DataLoader(dataset(is_train=True),batch_size=batch_size,shuffle=False)
    test_input=test_in()
    net.cuda()
    
#%%
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=0.001)
    criterion = nn.MSELoss(reduction='mean')
    z=0
    gain=[]
    pred=[]
    for i in range(epochs):
        z=0
        for num_batch,data in enumerate(trainloader,0):
            net.train()
            inputs,label=data

            inputs,label=inputs.float().cuda(),label.float().cuda()
            out=net(inputs)
            
            loss=MSE(torch.squeeze(out),label)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            z+=loss.item()
            
        print('train_loss= %.2f,  %d epoch left'%((z/(num_batch+1)),epochs-i))
        
        if (i %50)==0:
            last_price=0
            flag=0
            gain=[]
            with open(args.output, 'w', newline='') as csvfile:
                for j in range(len(test_data_csv)-1):
                    with torch.no_grad():
                        net.train()
                        test_input=test_input.float().cuda()
                        test_out=net(test_input)
                        test_out=torch.squeeze(test_out)
                        for k in range(num_day):
                            if k < num_day-2:
                                test_input[0][k]=test_input[0][k+1]
                            else:
                                test_input[0][k]=torch.from_numpy(test_data[j]).float().cuda()

                        if(flag==1 or flag==-1 ):
                            if(flag==1):
                                if test_out > last_price:
                                    writer = csv.writer(csvfile)
                                    writer.writerow(['-1'])
                                    gain.append(-1)
                                    flag=0
                                else:
                                    if(j==(len(test_data_csv)-2)):
                                        writer = csv.writer(csvfile)
                                        writer.writerow(['-1'])
                                        gain.append(-1)
                                    else:
                                        writer = csv.writer(csvfile)
                                        writer.writerow(['0'])
                                        gain.append(0)
                            else:
                                if test_out < last_price:
                                    writer = csv.writer(csvfile)
                                    writer.writerow(['1'])
                                    gain.append(1)
                                    flag=0
                                else:
                                    if(j==(len(test_data_csv)-2)):
                                        writer = csv.writer(csvfile)
                                        writer.writerow(['1'])
                                        gain.append(1)
                                    else:
                                        writer = csv.writer(csvfile)
                                        writer.writerow(['0'])
                                        gain.append(0)
                        else:
                            if(j!=(len(test_data_csv)-2)):
                                randn=random.randint(1, 2)
                                if(randn==2):
                                    randn=-1
                                flag=randn
                                last_price=test_out

                                writer = csv.writer(csvfile)
                                writer.writerow([flag])
                                gain.append(flag)

                            else:
                                writer = csv.writer(csvfile)
                                writer.writerow(['0'])
                                gain.append(0)
                flag=0
                save=0
                money=0
                for v in range(len(test_data_csv)-1):
                    if flag ==0:
                        flag=gain[v]
                        save=test_z[v][0]
                    elif gain[v]!=0 and v !=0:
                        if flag==(-1):
                            money+=save-test_z[v][0]
                            flag=0
                        else:
                            money+=test_z[v][0]-save
                            flag=0
                
                print('%.2f'%money)

                     
