import numpy as np
import torch
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
train_batch_size = 100
test_batch_size = 50
num_epoches = 60

path = 'matlab.mat'
data = scio.loadmat(path)
beardata=data['traindata'] 
beardatalab = data['traindatalab']

# for i in range(10):
#     print(beardatalab[4700*i])

beardata0=np.repeat(beardata[0:4700],3,0)
beardatalab0=np.repeat(beardatalab[0:4700],3,0)

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



# beardata=np.concatenate([beardata0,beardata1,beardata3,beardata4,beardata5,beardata6,beardata7,beardata8,beardata9,beardata2])
# beardatalab=np.concatenate([beardatalab0,beardatalab1,beardatalab3,beardatalab4,beardatalab5,beardatalab6,beardatalab7,beardatalab8,beardatalab9,beardatalab2])

print(beardata.shape)

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


for i in range(beardata.shape[0]):
    if(beardatalab[i]==7): 
        beardatalab[i]=2
    elif(beardatalab[i]!=0 and beardatalab[i]!=1):
        beardatalab[i]=3

print('Counter(data)\n',Counter(beardatalab)) # 调用Counter函数
print('==========')

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

# flaglab=0
# for img, label in beartrain_loader:
#     if(flaglab<4):
#         for i in range(img.shape[0]):
#             # print(img[i])
#             if(flaglab==label[i]):
#                 plt.subplot(10,1,flaglab+1)
#                 plt.title(label[i].numpy())
#                 plt.plot(img[i].numpy())
#                 flaglab+=1
#     else:
#         plt.show()
#         break
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
            nn.Linear(64*445,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10))       
             
        self.b2Layer1= nn.Sequential(
            nn.MaxPool1d(kernel_size=8,stride=2))
        self.b2Layer2= nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(7944,4),
            nn.ReLU(inplace=True),
            nn.Linear(4,4))
    def get_label_change(self,label_in,branch):
        label=label_in
        # print(l)
        # if (branch==1):
        #     for i in range(label.shape[0]):
        #         if(label[i]==7): 
        #             label[i]=2
        #         elif(label[i]!=0 and label[i]!=1):
        #             label[i]=3
        # print('Counter(data)\n',Counter(label.cpu().numpy())) # 调用Counter函数
        # print('==========')
        return label
    # def branch_crossEntropyLoss(self,out,label,barnnch)
    def get_forward_loss(self,x,label,branch):
        label=self.get_label_change(label,branch)
        criterion = nn.CrossEntropyLoss()
        out = self.forward(x,branch)
      

        return out,criterion(out, label)
        # out= F.softmax(out,dim=1)
        # loss1=0
        # if(branch==1):        
        #     for i in range(label.shape[0]):
        #         # print(out[i][label][i])    
        #         # loss1+=-torch.log(out[i][label[i]])    
        #         if(label[i]!=3): 
        #             loss1+=-torch.log(out[i][label[i]])*5.0/3.0
        #         else:
        #             loss1+=-torch.log(out[i][label[i]])*5.0/7.0      
        #     loss1/=label.shape[0]
        # return out,loss1
        

    def forward(self,x,branch):
        x = self.layer1(x)
        if(branch==2):            
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            # print(x.shape)   
            x = x.view(x.size(0),-1) #第二次卷积的输出拉伸为一行
            # print(x.shape)       
            x = self.fc(x)
        elif(branch==1):
            # print(label)
           
            x=self.b2Layer1(x)
            x = x.view(x.size(0),-1)
            x=self.b2Layer2(x)
            # x=F.softmax(x,dim=1)
        return x 
   
# lr = 0.0002 不均衡收敛
lr = 0.00015
momentum = 0.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model =BranchCNN()
# model.to(device)
model.to(device)
branch=1

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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
print('平安喜乐')
for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()
    #动态修改参数学习率
    if epoch%5==0:
        optimizer.param_groups[0]['lr']*=0.9
        print(optimizer.param_groups[0]['lr'])
    for img, label in train_loader:
        img=img.to(device)
        label = label.to(device)
        label=label.long()
        # img = img.view(img.size(0), -1)
        # 前向传播
        img = img.view(img.size(0),1,2000)
        out,loss=model.get_forward_loss(img,label,branch)
        # out = model.forward(img,branch)

        
        #out2 = model.forward(img,branch=2)
        # print(out[0])
        # print(label[0])
        # loss = criterion(out, label)#+criterion(out2, label)
        # loss=0
        # out= F.softmax(out,dim=1)
        # for i in range(label.shape[0]):
        #     # print(out[i][label][i])           
        #     loss+=-torch.log(out[i][label[i]])
        # loss/=label.shape[0]
        # print(loss.shape)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 保存loss的数据与epoch数值
        writer.add_scalar('Train', train_loss/len(train_loader), epoch)
        # 计算分类的准确率
        _, pred = out.max(1)

        num_correct = (pred == model.get_label_change(label,branch)).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    # 在测试集上检验效果
    eval_loss = 0   
    eval_acc = 0
    #net.eval() # 将模型改为预测模式
    model.eval()
    
    eval_acces_class_num=[0]*4
    eval_acces_class_num_correct=[0]*4
    for img, label_in in test_loader: 
        label=model.get_label_change(label_in,branch)
        # print(label)
        img=img.to(device)
        label = label.to(device)
        label = label.long()
        img = img.view(img.size(0),1,2000)
        # img = img.view(img.size(0), -1)
        # t=time.time()
        #out = model.forward(img,branch)
        # print(time.time()-t)
        # out2=model.forward(img,branch=2)
        out,loss = model.get_forward_loss(img,label,branch) 
        # criterion(out, label)
       # eval_losses_class[label]
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

        for i in range(label.shape[0]):
            # print(label.shape[0])
            eval_acces_class_num[label[i]]+=1
            if(pred[i]==label[i]):
                eval_acces_class_num_correct[label[i]]+=1
            else:
                eval_acces_class_num[pred[i]]+=1
        #eval_acces_class+=acc_class

        
    eval_acces_class=np.array(eval_acces_class_num_correct)/np.array(eval_acces_class_num)

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader), 
                     eval_loss / len(test_loader), eval_acc / len(test_loader)))
    #np.sort(eval_acces_class)
    for i in range(4):
        maxsite=np.argmax(eval_acces_class)
        print('class {},acc: {:.4f}'.format(maxsite,eval_acces_class[maxsite]))
        eval_acces_class[maxsite]=-1
    # for i in range(10):
    #     print('class {},acc: {:.4f}'.format(i,eval_acces_class[i]))
    

