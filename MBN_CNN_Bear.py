import numpy as np
import torch
from torchvision.datasets import mnist 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import scipy.io as scio
import time
from collections import Counter
import logging, os

train_batch_size = 100
test_batch_size = 1
num_epoches = 60


LOG_FILE = 'MBN_CNN_Bear.txt'
LOG_FORMAT = "%(message)s"

class Log():
    def __init__(self, clean = False):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(LOG_FORMAT)

        if clean:
            if os.path.isfile(LOG_FILE):
                with open(LOG_FILE, 'w') as f:
                    pass

        fh = logging.FileHandler(LOG_FILE)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def log(self, *args):
        s = ''
        for i in args:
            s += (str(i) + ' ')

        logging.debug(s)

log = Log(True)

path = 'matlab.mat'
data = scio.loadmat(path)
beardata=data['traindata'] 
beardatalab = data['traindatalab']
# log_file = open('MBN_bear成功版本.txt', 'w')

# for i in range(10):
#     log.log(beardatalab[4700*i])

beardata0=np.repeat(beardata[0:4700],10,0)
beardatalab0=np.repeat(beardatalab[0:4700],10,0)

beardata1=np.repeat(beardata[4700:2*4700],3,0)
beardatalab1=np.repeat(beardatalab[4700:2*4700],3,0)

beardata7=np.repeat(beardata[7*4700:8*4700],3,0)
beardatalab7=np.repeat(beardatalab[7*4700:8*4700],3,0)


beardata2=beardata[2*4700:3*4700-2350]
beardatalab2=beardatalab[2*4700:3*4700-2350]
beardata3=beardata[3*4700:4*4700-2350]
beardatalab3=beardatalab[3*4700:4*4700-2350]
beardata4=beardata[4*4700:5*4700-2350]
beardatalab4=beardatalab[4*4700:5*4700-2350]
beardata5=beardata[5*4700:6*4700-2350]
beardatalab5=beardatalab[5*4700:6*4700-2350]
beardata6=beardata[6*4700:7*4700-2350]
beardatalab6=beardatalab[6*4700:7*4700-2350]
beardata8=beardata[8*4700:9*4700-2350]
beardatalab8=beardatalab[8*4700:9*4700-2350]
beardata9=beardata[9*4700:10*4700-2350]
beardatalab9=beardatalab[9*4700:10*4700-2350]

beardata=np.concatenate([beardata,beardata0])
beardatalab=np.concatenate([beardatalab,beardatalab0])
# beardata=np.concatenate([beardata0,beardata1,beardata3,beardata4,beardata5,beardata6,beardata7,beardata8,beardata9,beardata2])
# beardatalab=np.concatenate([beardatalab0,beardatalab1,beardatalab3,beardatalab4,beardatalab5,beardatalab6,beardatalab7,beardatalab8,beardatalab9,beardatalab2])

log.log(beardata.shape)

index = [i for i in range(len(beardata))] 
np.random.shuffle(index)
beardata = beardata[index]
beardatalabtmp = beardatalab[index]

MAX=(np.amax(beardata))
MIN=(np.amin(beardata))
beardatalab=np.zeros(beardata.shape[0])
for i in range(beardata.shape[0]):
    beardatalab[i]=beardatalabtmp[i][0]


beardata=(((beardata-MIN)/(MAX-MIN)-0.5)*2).astype(np.float32)


# for i in range(beardata.shape[0]):
#     if(beardatalab[i]==7): 
#         beardatalab[i]=2
#     elif(beardatalab[i]!=0 and beardatalab[i]!=1)               :
#         beardatalab[i]=3

log.log('Counter(data)\n',Counter(beardatalab)) # 调用Counter函数
log.log('==========')
# log_file.close()
beartraindata=beardata[0:int(beardata.shape[0]/10)*8]
beartraindatalab=beardatalab[0:int(beardata.shape[0]/10)*8]
beartestdata=beardata[int(beardata.shape[0]/10)*8:int(beardata.shape[0]/10)*10]
beartestdatalab=beardatalab[int(beardata.shape[0]/10)*8:int(beardata.shape[0]/10)*10]




beartraindataset=TensorDataset(torch.from_numpy(beartraindata),torch.from_numpy(beartraindatalab))
beartestdataset=TensorDataset(torch.from_numpy(beartestdata),torch.from_numpy(beartestdatalab))

beartrain_loader = DataLoader(beartraindataset, batch_size=train_batch_size, shuffle=True)
beartest_loader = DataLoader(beartestdataset, batch_size=test_batch_size, shuffle=False)
train_loader=beartrain_loader
test_loader = beartest_loader

class BranchCNN(nn.Module):
    def __init__(self):
        super(BranchCNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,8,kernel_size=9),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv1d(8,16,kernel_size=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4,stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(16,32,kernel_size=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv1d(32,64,kernel_size=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4,stride=2))        
        self.fc = nn.Sequential(
            nn.Linear(64*445,128),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024,64),
            nn.ReLU(inplace=True),
            nn.Linear(128,4))       
             
        self.b1Layer1= nn.Sequential(
            nn.MaxPool1d(kernel_size=8,stride=2))
        self.b1Layer2= nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(7944,4),
            nn.ReLU(inplace=True),
            nn.Linear(4,4))
        self.b2Layer1= nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(15792,5),
            nn.ReLU(inplace=True),
            nn.Linear(5,5))   
    def get_label_change(self,label_in1,branch):
        label_in=label_in1.clone()
        # log.log(l)  
        if (branch==1):
            for i in range(label_in.shape[0]):
                if(label_in[i]==7): 
                    label_in[i]=2
                elif(label_in[i]!=0 and label_in[i]!=1):
                    label_in[i]=3
        elif(branch==2):
            for i in range(label_in.shape[0]):
                if(label_in[i]==3): 
                    label_in[i]=0
                elif(label_in[i]==5):
                    label_in[i]=1
                elif(label_in[i]==8):
                    label_in[i]=2
                elif(label_in[i]==9):
                    label_in[i]=3
                else:
                    label_in[i]=4
        elif(branch==0):
            for i in range(label_in.shape[0]):
                if(label_in[i]==2): 
                    label_in[i]=0
                elif(label_in[i]==4):
                    label_in[i]=1
                elif(label_in[i]==6):
                    label_in[i]=2
                else:
                    label_in[i]=3
        return label_in
    # def branch_crossEntropyLoss(self,out,label,barnnch)
    def forward_result(self,x):
        x1 = self.layer1(x)
        # log.log(label)           
        x=self.b1Layer1(x1)
        x = x.view(x.size(0),-1)
        x=self.b1Layer2(x)
        # x=F.softmax(x,dim=1)        
        pre=x[0].argmax()
        if(pre==0):
            return 0
        elif(pre==1):
            return 1
        elif(pre==2):
            return 7
        x1=self.layer2(x1)        
        x= x1.view(x1.size(0),-1)
        x=self.b2Layer1(x)
        pre=x[0].argmax()
        if(pre==0):
            return 3
        elif(pre==1):
            return 5
        elif(pre==2):
            return 8
        elif(pre==3):
            return 9
        x = self.layer3(x1)
        x = self.layer4(x)
        # log.log(x.shape)   
        x = x.view(x.size(0),-1) #第二次卷积的输出拉伸为一行
        # log.log(x.shape)       
        x = self.fc(x)             
        pre=x[0].argmax()    
        if(pre==0):
            return 2
        elif(pre==1):
            return 4
        elif(pre==2):
            return 6
        else:
            return 0

    def get_forward_loss(self,x,label_in):
        
        criterion = nn.CrossEntropyLoss()
        out1 = self.forward(x,1)
        out2 = self.forward(x,2)
        out0 = self.forward(x,0)
        out=out0[:,0]
        # log.log(out.shape)
        for i in range(out1.shape[0]):
            pre=out1[i].argmax()
            # log.log(out1[i])
            # log.log(out1[i].max(1))
            
            if(pre==0):
                out[i]=0
            elif(pre==1):
                out[i]=1
            elif(pre==2):
                out[i]=7
            pre=out2[i].argmax()
            if(pre==0):
                out[i]=3
            elif(pre==1):
                out[i]=5
            elif(pre==2):
                out[i]=8
            elif(pre==3):
                out[i]=9 
            pre=out0[i].argmax()
            if(pre==0):
                out[i]=2
            elif(pre==1):
                out[i]=4
            elif(pre==2):
                out[i]=6
            
            # else:
            #     out[i]=0

        # log.log('Counter(data)\n',Counter(label_tmp.cpu().numpy())) # 调用Counter函数
        # log.log('==========')

        # log.log(label_tmp.shape)
        # log.log(out.shape)
        return out,criterion(out1, self.get_label_change(label_in,1))+\
            criterion(out2, self.get_label_change(label_in,2))+\
            criterion(out0, self.get_label_change(label_in,0))
        # out= F.softmax(out,dim=1)
        # loss1=0
        # if(branch==1):        
        #     for i in range(label_tmp.shape[0]):
        #         # log.log(out[i][label][i])    
        #         # loss1+=-torch.log(out[i][label[i]])    
        #         if(label_tmp[i]!=3): 
        #             loss1+=-torch.log(out[i][label_tmp[i]])*5.0/3.0
        #         else:
        #             loss1+=-torch.log(out[i][label_tmp[i]])*5.0/7.0      
        #     loss1/=label_tmp.shape[0]
        # return out,loss1
        

    def forward(self,x,branch):
        x = self.layer1(x)
        if(branch==1):
            # log.log(label)           
            x=self.b1Layer1(x)
            x = x.view(x.size(0),-1)
            x=self.b1Layer2(x)
            # x=F.softmax(x,dim=1)
        else:
            x=self.layer2(x)
            if(branch==2):
                x= x.view(x.size(0),-1)
                x=self.b2Layer1(x)
            elif(branch==0):
                x = self.layer3(x)
                x = self.layer4(x)
                # log.log(x.shape)   
                x = x.view(x.size(0),-1) #第二次卷积的输出拉伸为一行
                # log.log(x.shape)       
                x = self.fc(x)       
        return x 
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,8,kernel_size=16),
            # nn.Conv2d(1,16,kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv1d(8,16,kernel_size=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4,stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(16,32,kernel_size=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv1d(32,64,kernel_size=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4,stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(28416,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10))            
        self.b2Layer1= nn.Sequential(
            nn.Conv1d(1,8,kernel_size=8),
            # nn.Conv2d(1,16,kernel_size=3),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8,stride=2))
            # nn.Linear(8*1992,10))
        self.b2Layer2= nn.Sequential(
            # nn.MaxPool1d(kernel_size=16,stride=2),
            nn.Linear(7944,10))
            # nn.ReLU(inplace=True),
            # nn.Linear(10,10))
    def forward(self,x,branch):
        
        if(branch==1):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            # print(x.shape)   
            x = x.view(x.size(0),-1) #第二次卷积的输出拉伸为一行
            # print(x.shape)       
            x = self.fc(x)
        
        elif(branch==2):
            # print(x.shape)
           
            x=self.b2Layer1(x)
            x = x.view(x.size(0),-1)
            x=self.b2Layer2(x)
            
        return x


#实例化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs")
#    # dim = 0 [20, xxx] -> [10, ...], [10, ...] on 2GPUs
#    model = nn.DataParallel(model)
model =CNN()
model.to(device)

model_MBN =BranchCNN()
model_MBN.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

losses = []
acces = []
eval_losses = []
eval_acces = []
eval_losses_class = []
eval_acces_class = []
writer = SummaryWriter(log_dir='logs',comment='train-loss')
# input = torch.rand(32,1,28,28)
# with SummaryWriter(log_dir='logs',comment='Net') as w:
#     w.add_graph(model,(input,))
log.log('平安喜乐')


#net.eval() # 将模型改为预测模式
model.load_state_dict(torch.load('CNN_bearmodel.pkl'))
model_MBN.load_state_dict(torch.load('MBN_bear成功版本model.pkl'))
model_MBN.eval()
model.eval()

eval_acces_class_num=[0]*10
eval_acces_class_num_correct=[0]*10
cntbatch=0
tcnn=0
tmbn=0
totalcnn=0
totalmbn=0
for img, label in test_loader: 
    img=img.to(device)
    img = img.view(img.size(0),1,2000)
    # img = img.view(img.size(0), -1)
    t=time.time()
    out = model.forward(img,1)
    label_cnn=out.argmax()
    tcnn=time.time()-t
    totalcnn+=tcnn
    t=time.time()
    out1=model_MBN.forward_result(img)    
    tmbn=time.time()-t
    totalmbn+=tmbn
    log.log('')
    # print(label_cnn.shape)
    # print(out1.shape)
    # 记录误差
      # 记录准确率
    
    cntbatch+=1
    log.log('batch num: %d/%d,label=%d %d %d,tcnn=%f,totalcnn=%f,tmbn=%f,totalmbn=%f'%(cntbatch,len(beartestdataset)/test_batch_size,label[0],label_cnn,out1,tcnn,totalcnn,tmbn,totalmbn))


