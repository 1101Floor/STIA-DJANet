import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#评价函数
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import networkx as nx
import itertools
import random
from tqdm import trange
import warnings
import math
from math import *

import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset,DataLoader
import random  
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
warnings.filterwarnings('ignore')
data = pd.read_csv('##########')#dataset
train_data = []
val_data = []
test_data = []

train_val = 0.7
train_test_cut = 0.8
for i in np.unique(data['mark']):
    df = data.loc[data['mark']==i].iloc[:,[0,1,2,5,6,7,8,9]]    
    cut_train = df.loc[df['mark']==i].iloc[:int(len(df)*train_val),:].sort_values(by=['UnixTime'],ascending=True)
    train_data.extend(cut_train.values)
    
    cut_val = df.loc[df['mark']==i].iloc[int(len(df)*train_val):int(len(df)*train_test_cut),:].sort_values(by=['UnixTime'],ascending=True)
    val_data.extend(cut_val.values)
    
    cut_test= df.loc[df['mark']==i].iloc[int(len(df)*train_test_cut):,:].sort_values(by=['UnixTime'],ascending=True)
    test_data.extend(cut_test.values)
train_data_or = pd.DataFrame(train_data,columns = ['UnixTime','Lon','Lat','SOG','COG','mark','Length','Width'])
val_data_or = pd.DataFrame(val_data,columns = ['UnixTime','Lon','Lat','SOG','COG','mark','Length','Width'])
test_data_or = pd.DataFrame(test_data,columns = ['UnixTime','Lon','Lat','SOG','COG','mark','Length','Width'])

train_data_mm = pd.DataFrame(train_data,columns = ['UnixTime','Lon','Lat','SOG','COG','mark','Length','Width'])
val_data_mm = pd.DataFrame(val_data,columns = ['UnixTime','Lon','Lat','SOG','COG','mark','Length','Width'])
test_data_mm = pd.DataFrame(test_data,columns = ['UnixTime','Lon','Lat','SOG','COG','mark','Length','Width'])

scaler_minmax = MinMaxScaler()
features = ['Lon', 'Lat', 'SOG', 'COG','Length','Width']
scaler_minmax.fit(train_data_mm[features])
train_data_mm[features] = scaler_minmax.transform(train_data_mm[features])
val_data_mm[features] = scaler_minmax.transform(val_data_mm[features])
test_data_mm[features] = scaler_minmax.transform(test_data_mm[features])
#######label
train_data_label = []
val_data_label = []
test_data_label = []

train_val = 0.7
train_test_cut = 0.8
i=83
df_label = data.loc[data['mark']==i].iloc[:,[0,1,2,7]]    
cut_train_label = df_label.loc[df_label['mark']==i].iloc[:int(len(df_label)*train_val),:].sort_values(by=['UnixTime'],ascending=True)
cut_val_label = df_label.loc[df_label['mark']==i].iloc[int(len(df_label)*train_val):int(len(df_label)*train_test_cut),:].sort_values(by=['UnixTime'],ascending=True)
cut_test_label= df_label.loc[df_label['mark']==i].iloc[int(len(df_label)*train_test_cut):,:].sort_values(by=['UnixTime'],ascending=True)

label_scaler_minmax = MinMaxScaler()
label_features = ['LON', 'LAT']
label_scaler_minmax.fit(cut_train_label[label_features])
cut_train_label[label_features] = label_scaler_minmax.transform(cut_train_label[label_features])
cut_val_label[label_features] = label_scaler_minmax.transform(cut_val_label[label_features])
cut_test_label[label_features] = label_scaler_minmax.transform(cut_test_label[label_features])
#data_train = data.iloc[0:int(len(data)),[1,2,5,6]]
print('train_data',train_data_mm.shape)
print('val_data',val_data_mm.shape)
print('test_data',test_data_mm.shape)
class tarship():
    def __init__(self, lat, lon, cog, sog):
        self.lat = lat
        self.lon = lon
        self.cog = cog
        self.sog = sog


class refship():
    def __init__(self, lat, lon, cog, sog):
        self.lat = lat
        self.lon = lon
        self.cog = cog
        self.sog = sog
        
class Cal():
    def __init__(self, tar_ship, ref_ship):
        self.tar_lat = tar_ship.lat
        self.tar_lon = tar_ship.lon
        self.tar_cog = tar_ship.cog
        self.tar_sog = tar_ship.sog
        self.ref_lat = ref_ship.lat
        self.ref_lon = ref_ship.lon
        self.ref_cog = ref_ship.cog
        self.ref_sog = ref_ship.sog
        self.differ_lon = self.tar_lon - self.ref_lon
        self.differ_cog = self.ref_cog - self.tar_cog
        self.differ_lon2 = self.tar_lon - self.ref_lon
        self.ref_lat2 = ref_ship.lat 

    def dist(self):
        if self.ref_lat >= 0 and self.ref_lat * self.tar_lat >= 0:  
            self.ref_lat = self.ref_lat
            self.tar_lat = self.tar_lat
        elif self.ref_lat >= 0 and self.ref_lat * self.tar_lat < 0: 
            self.ref_lat = self.ref_lat
            self.tar_lat = self.tar_lat
        elif self.ref_lat < 0 and self.ref_lat * self.tar_lat >= 0:
            self.tar_lat = -self.tar_lat
            self.ref_lat = -self.ref_lat
        elif self.ref_lat < 0 and self.ref_lat * self.tar_lat < 0:
            self.tar_lat = -self.tar_lat
            self.ref_lat = -self.ref_lat
        if fabs(self.differ_lon) >= 180:  
            self.differ_lon = 360 - fabs(self.differ_lon)
        D = acos(sin(radians(self.tar_lat)) * sin(radians(self.ref_lat)) + cos(radians(self.tar_lat)) * cos(
            radians(self.ref_lat)) * cos(radians(fabs(self.differ_lon))))  
        return D * 180 / pi * 60

    def true_bearing(self):
        if self.ref_lat >= 0 and self.ref_lat * self.tar_lat >= 0:  #
            self.ref_lat = self.ref_lat
            self.tar_lat = self.tar_lat
        elif self.ref_lat >= 0 and self.ref_lat * self.tar_lat < 0:
            self.ref_lat = self.ref_lat
            self.tar_lat = self.tar_lat
        elif self.ref_lat < 0 and self.ref_lat * self.tar_lat >= 0:
            self.tar_lat = -self.tar_lat
            self.ref_lat = -self.ref_lat
        elif self.ref_lat < 0 and self.ref_lat * self.tar_lat < 0:
            self.tar_lat = -self.tar_lat
            self.ref_lat = -self.ref_lat
        if fabs(self.differ_lon) >= 180:  
            self.differ_lon = 360 - fabs(self.differ_lon)
        TB = 0
        if self.differ_lon == 0 or self.differ_lon == 180: 
            if self.ref_lat > self.tar_lat:
                p = 180
            else:
                p = 0
        else:
            a = tan(radians(self.tar_lat)) * cos(radians(self.ref_lat)) * 1 / sin(radians(fabs(self.differ_lon))) - sin(
                radians(self.ref_lat)) * 1 / tan(
                radians(fabs(self.differ_lon)))
            if a == 0:
                a = 0.00001
            p = (atan(1 / a)) * 180 / pi
        if self.differ_lon2 > 180: 
            self.differ_lon2 = -(360 - self.differ_lon2)
        elif self.differ_lon2 < -180:
            self.differ_lon2 = (360 + self.differ_lon2)
        if self.ref_lat2 >= 0:
            if self.differ_lon2 >= 0:
                if p > 0:
                    TB = p
                elif p < 0:
                    TB = 180 + p
            elif self.differ_lon2 < 0:
                if p > 0:
                    TB = 360 - p
                elif p < 0:
                    TB = 180 - p
        elif self.ref_lat2 < 0:
            if self.differ_lon2 >= 0:
                if p > 0:
                    TB = 180 - p
                elif p < 0:
                    TB = -p
            elif self.differ_lon2 < 0:
                if p > 0:
                    TB = 180 - p
                else:
                    TB = 360 - fabs(p)
        # print("TB:%s"%TB)
        # print("方位角为：%s" % (TB))
        return TB

    def cal_dcpa(self):
        if self.differ_cog >= 0:
            b = self.differ_cog
        else:
            b = 360 + self.differ_cog
        # print("b:%s"%b)
        a = self.tar_sog * self.tar_sog + pow(self.ref_sog, 2) - 2 * self.ref_sog * self.tar_sog * cos(radians(b))
        TB = self.true_bearing()
        d = fabs(TB - self.ref_cog)
        if d <= 180:
            Q = d
        elif d > 180:
            Q = 360 - d
        vx = sqrt(a)
        if self.ref_sog == 0 or self.tar_sog == 0:
            self.ref_sog = 0.001
            self.tar_sog = 0.001
        if vx < 0.00001: 
            vx = 0.0000001
        f = (pow(vx, 2) + pow(self.ref_sog, 2) - pow(self.tar_sog, 2)) / (2 * self.ref_sog * vx)

        alpha = acos(f) * 180 / pi
        D = self.dist()

        if b == 0: 
            dcpa = D
            tcpa = 0
        elif b <= 180:
            dcpa = D * sin(radians(fabs(Q - alpha)))
            tcpa = D * cos(radians(Q - alpha)) / vx
        else:
            dcpa = D * sin(radians(fabs(Q + alpha)))
            tcpa = D * cos(radians(Q + alpha)) / vx

        if vx < 0.000001:
            tcpa = 0
        elif vx < 0.000001 and self.ref_sog < 0.0001:
            tcpa = 10000000
        return dcpa
    def calculate_bearing(self):
        lat1_rad = math.radians(self.tar_lat)
        lon1_rad = math.radians(self.tar_lon)
        lat2_rad = math.radians(self.ref_lat)
        lon2_rad = math.radians(self.ref_lon)
        d_lon = lon2_rad - lon1_rad
        y = math.sin(d_lon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lon)
        initial_bearing = math.atan2(y, x)

        bearing = (math.degrees(initial_bearing) + 360) % 360

        return bearing
def convert_data(Len_his,Len_fut,data_in,data_out,test_ratio,val_ratio):
    data_in = np.array(data_in)
    data_out = np.array(data_out)
    cut1 = round(test_ratio* data_in.shape[0])
    model_input = []
    model_output = []
    for i in range(len(data_in)-Len_his-Len_fut+1):
        model_input.append(data_in[i:i+Len_his,:])
        model_output.append(data_out[i+Len_his:i+Len_his+Len_fut,:])
    model_input=np.array(model_input)
    model_output=np.array(model_output)  
    x_train,y_train,x_test,y_test=\
    model_input[:-cut1,:,:],model_output[:-cut1:],model_input[-cut1:,:,:],model_output[-cut1:]
    cut2 = round(val_ratio* x_train.shape[0])
    x_train,y_train,x_val,y_val=\
    x_train[:-cut2,:,:],y_train[:-cut2:],x_train[-cut2:,:,:],y_train[-cut2:]
    print('x_train.shape',x_train.shape)
    print('x_val.shape',x_val.shape)
    print('x_test.shape',x_test.shape)
    print('y_train.shape',y_train.shape)
    print('y_val.shape',y_val.shape)
    print('y_test.shape',y_test.shape)
    return x_train, x_val, x_test, y_train, y_val, y_test 
def load_dataset(Len_his,Len_fut,data,ref_mark,data_mm,label):
    mark_refship = np.unique(data['mark'])[0]
    mark_tarship = np.unique(data['mark'])[1:]
    vId = np.unique(data['mark'])[np.unique(data['mark'])!=ref_mark]
    ref_mark = ref_mark
    ref_data = np.asarray(data)[np.asarray(data)[:,-3]==ref_mark,:]
    dis = np.zeros((len(ref_data),len(np.unique(data['mark']))-1))
    for i in range(len(dis)):
        Y = ref_data 
        for j in range(len(vId)):
            tar_data = np.asarray(data)[np.asarray(data)[:,-3]==vId[j],:]
            tar = tarship(tar_data[i,2],tar_data[i,1],tar_data[i,4],tar_data[i,3])
            ref = refship(ref_data[i,2],ref_data[i,1],ref_data[i,4],ref_data[i,3])
            d = Cal(tar,ref)
            dis[i][j]=d.cal_dcpa()
    bear = np.zeros((len(ref_data),len(np.unique(data['mark']))-1))
    for i in range(len(dis)):
        Y = ref_data 
        for j in range(len(vId)):
            tar_data = np.asarray(data)[np.asarray(data)[:,-3]==vId[j],:]
            tar = tarship(tar_data[i,2],tar_data[i,1],tar_data[i,4],tar_data[i,3])
            ref = refship(ref_data[i,2],ref_data[i,1],ref_data[i,4],ref_data[i,3])
            d = Cal(tar,ref)
            bear[i][j]=d.calculate_bearing()
    bear_1 = np.zeros((len(ref_data),len(np.unique(data['mark']))-1), dtype=int) 
    for i in range(len(bear)):
        for j in range(len(vId)):
            if bear[i][j]>=22.5 and bear[i][j]<67.5:
                bear_1[i][j]=2   
            elif bear[i][j]>=67.5 and bear[i][j]<112.5:
                bear_1[i][j]=3
            elif bear[i][j]>=112.5 and bear[i][j]<157.5:
                bear_1[i][j]=4
            elif bear[i][j]>=157.5 and bear[i][j]<202.5:
                bear_1[i][j]=5
            elif bear[i][j]>=202.5 and bear[i][j]<247.5:
                bear_1[i][j]=6
            elif bear[i][j]>=247.5 and bear[i][j]<292.5:
                bear_1[i][j]=7
            elif bear[i][j]>=292.5 and bear[i][j]<337.5:
                bear_1[i][j]=8
            else:
                bear_1[i][j]=1
    ref_data_1 = np.asarray(data_mm)[np.asarray(data_mm)[:,-3]==ref_mark,:]
    spin = np.zeros((len(ref_data_1),9,7)) 
    time = ref_data_1[:,0]
    for i in range(len(time)):
        ind_time = time[i]
        ind_bear = bear_1[i]
        unique_bear, counts = np.unique(bear_1[i], return_counts=True)  
        spin[i,0,:] = np.append(ref_data_1[[i],[1,2,3,4,6,7]],0)
        for j in range(len(vId)):
            if len(unique_bear)==len(vId) and dis[i][j]<=3:
                dcpa = dis[i][j]
                tar_data = np.asarray(data_mm)[np.asarray(data_mm)[:,-3]==vId[j],:]
                spin[i,bear_1[i][j],:] = np.append(tar_data[[i],[1,2,3,4,6,7]],dis[i][j])
            else:
                for idx in unique_bear:
                    positions = np.where(ind_bear == idx)[0]
                    if dis[i][j]<=3:
                        tar_data =np.asarray(data_mm)[np.isin(np.asarray(data_mm)[:, -3], vId[positions]), :]
                        tar_data = tar_data[tar_data[:,0]==ind_time,:]
                        spin[i,bear_1[i][j],:] = np.mean(np.hstack((tar_data[:,[1,2,3,4,6,7]],dis[i][positions].reshape(-1,1))), axis=0)
    label = np.asarray(label)[:,[1,2]]
    Ref = ref_data_1[:,1:5]
    VOI_in = []
    Ref_in = []
    out = []
    for i in range(len(label)-Len_his-Len_fut+1):
        VOI_in.append(spin[i:i+Len_his,:])
        Ref_in.append(Ref[i:i+Len_his,:])
        out.append(label[i+Len_his:i+Len_his+Len_fut,:])
    VOI_in = np.array(VOI_in)
    Ref_in = np.array(Ref_in)
    out = np.array(out)
    N = VOI_in[:,:,1,:]
    NE = VOI_in[:,:,2,:]
    E = VOI_in[:,:,3,:]
    SE = VOI_in[:,:,4,:]
    S = VOI_in[:,:,5,:]
    SW = VOI_in[:,:,6,:]
    W = VOI_in[:,:,7,:]
    NW = VOI_in[:,:,8,:]
    SELF = VOI_in[:,:,0,:]
    return Ref_in, N, NE, E, SE, S, SW, W, NW, SELF, out
class MyDataset(Dataset):
    def __init__(self, Ref_in, N, NE, E, SE, S, SW, W, NW, SELF, out):
        self.Ref_in = Ref_in
        self.N = N
        self.NE = NE
        self.E = E
        self.SE = SE
        self.S = S
        self.SW = SW
        self.W = W
        self.NW = NW
        self.SELF = SELF
        self.out = out
    def __getitem__(self, index):

        output = [self.Ref_in[index], self.N[index], self.NE[index], 
                  self.E[index], self.SE[index], self.S[index], self.SW[index],
                  self.W[index], self.NW[index], self.SELF[index], self.out[index]]
        return output
    def __len__(self):
        return self.Ref_in.shape[0]
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return x
class Cross_Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Cross_Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n
class DJAU(nn.Module):

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv1d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv1d(dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv1d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        # append a se operation
        b, c, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1)
        return se_atten * f_x * u 
class STIADJANet(nn.Module):
    def __init__(self):
        super(STIADJANet,self).__init__()

        self.HST_LEN,self.FUT_LEN=20,10
        self.self_input_size= 4
       
        self.ego_enc_size=64
        self.output_size = 2
        
        self.cross_input_size= 7
        self.ego_sur_size=32  
        
        self.encoder = Encoder(self.HST_LEN, self.self_input_size, self.ego_enc_size).to(device)
        self.cross_encoder = Cross_Encoder(self.HST_LEN, self.cross_input_size, self.ego_sur_size).to(device)

        self.conv3x3=torch.nn.Conv2d(self.ego_sur_size, self.ego_sur_size, (3,3))
        self.conv1x1=torch.nn.Conv2d(self.ego_sur_size, self.ego_sur_size, (1,1))
        
        self.op1=nn.Linear(self.ego_sur_size+self.ego_sur_size+self.ego_sur_size*9+self.ego_enc_size*self.HST_LEN*2, self.FUT_LEN*self.output_size)

        self.DJAU = DJAU(dim=20,kernel_size=3, dilation=3, reduction=3)

        self.leaky_relu=torch.nn.LeakyReLU(0.1)
        self.relu=torch.nn.ReLU()
        self.softmax=torch.nn.Softmax(dim=1)        
        self.maxpool = nn.MaxPool2d(2,stride=2)
        
    def forward(self,Self_in,N,NE,E,SE,S,SW,W,NW,SELF):

        self_aware = self.leaky_relu(self.encoder(Self_in))
        
        N_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(N)))
        NE_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(NE)))
        E_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(E)))
        SE_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(SE)))
        S_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(S)))
        SW_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(SW)))
        W_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(W)))
        NW_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(NW)))
        CE_enc=torch.squeeze(self.leaky_relu(self.cross_encoder(SELF)))
        a,b,c = SELF.size()[0],SELF.size()[1],SELF.size()[2]
        
        
        cos_T=torch.stack((NW_enc,N_enc,NE_enc),2).contiguous()
        cos_M=torch.stack((W_enc,CE_enc,E_enc),2).contiguous()
        cos_B=torch.stack((SW_enc,S_enc,SE_enc),2).contiguous()
        sur_cs=torch.stack((cos_T,cos_M,cos_B),2)
        
        sur_conv1x1=self.leaky_relu(self.conv1x1(sur_cs))
        sur_conv1x1_pool = self.leaky_relu(self.maxpool(self.conv1x1(sur_cs)))
        sur_conv3x3=self.leaky_relu(self.conv3x3(sur_cs))
        
        sur_conv1x1=torch.squeeze(sur_conv1x1.reshape(sur_conv1x1.shape[0],-1))
        sur_conv1x1_pool=torch.squeeze(sur_conv1x1_pool)
        sur_conv3x3=torch.squeeze(sur_conv3x3)
        sur_conv = torch.cat((sur_conv1x1,sur_conv1x1_pool,sur_conv3x3),1)
        DJAU=self.DJAU(self_aware)
        SW = DJAU+self_aware
        SW = SW.reshape(a,-1)

        self_aware_1 = self_aware.reshape(a,-1)
        enc = torch.cat((self_aware_1,SW,sur_conv),1)

        fut_pred=self.op1(enc)
        fut_pred = fut_pred.reshape(-1,self.FUT_LEN, 2)

        return fut_pred
device = torch.device("cuda")
Ref_in, N, NE, E, SE, S, SW, W, NW, SELF, out = load_dataset(20,10,train_data_or,target_mark,train_data_mm,cut_train_label)
dset_train = MyDataset(Ref_in, N, NE, E, SE, S, SW, W, NW, SELF, out)
Ref_in1, N1, NE1, E1, SE1, S1, SW1, W1, NW1, SELF1, out1 = load_dataset(20,10,val_data_or,target_mark,val_data_mm,cut_val_label)
dset_val = MyDataset(Ref_in1, N1, NE1, E1, SE1, S1, SW1, W1, NW1, SELF1, out1)
Ref_in2, N2, NE2, E2, SE2, S2, SW2, W2, NW2, SELF2, out2 = load_dataset(20,10,test_data_or,target_mark,test_data_mm,cut_test_label)
dset_test = MyDataset(Ref_in2, N2, NE2, E2, SE2, S2, SW2, W2, NW2, SELF2, out2)
loader_train = DataLoader(
        dset_train,
        batch_size=64, 
        shuffle =True,
        num_workers=0)

loader_val = DataLoader(
        dset_val,
        batch_size=64, 
        shuffle =True,
        num_workers=0)
loader_test = DataLoader(
        dset_test,
        batch_size=64, 
        shuffle =True,
        num_workers=0)
model = STIADJANet().to(device)
class ExponentialL2Loss(nn.Module):
    def __init__(self):
        super(ExponentialL2Loss, self).__init__()    
    def forward(self, predictions, targets, time_steps,gamma):
        l2_loss = torch.pow(predictions - targets, 2)
        if time_steps.dim() == 2:
            time_steps = time_steps.unsqueeze(-1)  # 扩展维度以便广播           
        decay_factor = torch.exp(time_steps/gamma).to(device)
        loss = l2_loss * decay_factor
        return loss.mean().to(device)
def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 2
    elif epoch > 150:
        lr = lr / 2
    elif epoch > 120:
        lr = lr / 2
    elif epoch > 90:
        lr = lr / 2
    elif epoch > 60:
        lr = lr / 2
    elif epoch > 30:
        lr = lr / 2

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
def test(model,dataloader):
    model.eval()
    total_loss = 0
    loss_function = ExponentialL2Loss()
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            batch = [tensor.cuda() for tensor in batch]
            Ref_in, N, NE, E, SE, S, SW, W, NW, SELF, out = batch
            
            result = model(Ref_in.float(), N.float(), NE.float(), E.float(), SE.float(), S.float(), SW.float(), W.float(), NW.float(), SELF.float())
            time_steps = torch.arange(result.shape[1]).expand(result.shape[0], result.shape[1]).float()
            loss = loss_function(result, out.float(), time_steps,5)+loss_fn(result, out.float())
            total_loss += loss
    return (total_loss/len(dataloader))
def train(model,train_dataloader,test_dataloader,num_epoch):
    train_loss_coll = []
    val_loss_coll = []
    min_loss = 1000
    loss_fn = torch.nn.MSELoss()
    loss_function = ExponentialL2Loss()
    for epoch in range(num_epoch):
        model = model.to(torch.float32)
        model.train()
        train_loss = 0

        for i,batch in enumerate(train_dataloader):
            batch = [tensor.cuda() for tensor in batch]
            Ref_in, N, NE, E, SE, S, SW, W, NW, SELF, out = batch
            optimizer.zero_grad()
            
            result = model(Ref_in.float(), N.float(), NE.float(), E.float(), SE.float(), S.float(), SW.float(), W.float(), NW.float(), SELF.float())
            time_steps = torch.arange(result.shape[1]).expand(result.shape[0], result.shape[1]).float()
            loss = loss_function(result, out.float(), time_steps,5)+loss_fn(result, out.float())
            loss.backward()
            optimizer.step()
            train_loss+=loss

        adjust_learning_rate(epoch)
        train_loss = train_loss/len(train_dataloader)
        test_loss = test(model,test_dataloader)
        torch.save(model,'CSC2.pt')

        train_loss_coll.append(train_loss)
        val_loss_coll.append(test_loss)

        print("Epoch {}, Train loss {}, val loss {}".format(epoch,train_loss,test_loss))

    return train_loss_coll, val_loss_coll

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
train_loss, val_loss = train(model, loader_train, loader_val,250)
epochs_range = range(len(train_loss))
plt.plot(epochs_range,[i.cpu().detach().numpy() for i in train_loss],label= "Train_loss")
plt.plot(epochs_range, [i.cpu().detach().numpy() for i in val_loss], label="Val_loss")
plt.legend(loc='upper right')
plt.title('Train and Val Loss')
plt.show()
def predict(model, test_dataloader):
    model.eval() 
    device = next(model.parameters()).device  
    tgts = []
    results = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            Ref_in, N, NE, E, SE, S, SW, W, NW, SELF, out = [x.to(device).float() for x in batch]
            
            result = model(Ref_in, N, NE, E, SE, S, SW, W, NW, SELF)
            tgts.extend(out.cpu().numpy()) 
            results.extend(result.cpu().numpy())
            
    tgts = torch.tensor(tgts) 
    results = torch.tensor(results)
    
    return tgts, results
#评价函数
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt

def mape(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape

def easy_result(y_train, y_train_predict, train_index, model_index, col_index):
  plt.figure(figsize=(10,5))
  plt.plot(y_train[:])
  plt.plot(y_train_predict[:])
  plt.legend(('real', 'predict'),fontsize='15')
  plt.title("%s Data"%train_index,fontsize='20')
  plt.show()
  print('\n')

  plot_begin,plot_end=min(min(y_train),min(y_train_predict)),max(max(y_train),max(y_train_predict))
  plot_x=np.linspace(plot_begin,plot_end,10)
  plt.figure(figsize=(5,5))
  plt.plot(plot_x,plot_x)
  plt.plot(y_train,y_train_predict,'o')
  plt.title("%s Data"%train_index,fontsize='20')
  plt.show()

  #输出结果
  print('%s上的MAE/RMSE/MAPE/R^2'%train_index)
  print(mean_absolute_error(y_train, y_train_predict))
  print(np.sqrt(mean_squared_error(y_train, y_train_predict) ))
  print(mape(y_train, y_train_predict) )
  print(r2_score(y_train, y_train_predict))
tgts , predicts= predict(model,loader_test)
tgts1 = tgts.reshape(-1,2)
predicts1= predicts.reshape(-1,2)
easy_result(tgts1[:,0], predicts1[:,0], 'Train', 'lstm', 'longitude')
easy_result(tgts1[:,1], predicts1[:,1], 'Train', 'lstm', 'latitude')
tgts_1 = label_scaler_minmax.inverse_transform(tgts1).reshape(-1,10,2)
predicts_1 = label_scaler_minmax.inverse_transform(predicts1).reshape(-1,10,2)
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math

def ls(lng, lat):
    lat = float(lat)
    lng = float(lng)
    x = lng * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return [float(x), float(y)]

def ll(x, y):
    x = float(x)
    y = float(y)
    x = x / 20037508.34 * 180
    y = y / 20037508.34 * 180
    y = 180 / math.pi * (2 * math.atan(math.exp(y * math.pi / 180)) - math.pi / 2)
    return [float(x), float(y)]

predict_traj = [np.array([ls(predicts_1[i, j, 0], predicts_1[i, j, 1]) for j in range(predicts_1.shape[1])]) for i in range(predicts_1.shape[0])]
target_traj = [np.array([ls(tgts_1[i, j, 0], tgts_1[i, j, 1]) for j in range(tgts_1.shape[1])]) for i in range(tgts_1.shape[0])]

def get_ade(predict_traj, target_traj):
    diffs = predict_traj - target_traj
    dist_squared = np.sum(diffs ** 2, axis=1)
    return np.mean(np.sqrt(dist_squared))

def get_fde(predict_traj, target_traj):
    diff = predict_traj[-1] - target_traj[-1]
    dist_squared = np.sum(diff ** 2)
    return np.sqrt(dist_squared)

adelist = [get_ade(predict_traj[i], target_traj[i]) for i in range(len(predict_traj))]
fdelist = [get_fde(predict_traj[i], target_traj[i]) for i in range(len(predict_traj))]

A = np.array(predict_traj).reshape(-1, 2)
B = np.array(target_traj).reshape(-1, 2)
MR = np.sqrt(np.sum((A - B) ** 2, axis=1))

print('mean_ade:{0},std_ade:{1}'.format(np.mean(adelist), np.std(adelist)))
print('mean_fde:{0},std_fde:{1}'.format(np.mean(fdelist), np.std(fdelist)))
print('MR@10:{0},MR@30:{1},MR@50:{2}'.format(1 - np.mean(MR <= 10), 1 - np.mean(MR <= 30), 1 - np.mean(MR <= 50)))