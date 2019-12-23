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
import logging, os
train_batch_size = 100
test_batch_size = 5
num_epoches = 60

LOG_FILE = 'MBN_bear成功版本log.txt'
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
def test():
    log.log('aaa')
    log.log('bbb', 3)
    log.log('ccc %.2f CCC' % (3.1415926))
    log.log('ddd', 'ee %.2f fff' % (3.1415926))

path = 'matlab.mat'
data = scio.loadmat(path)
beardata=data['traindata'] 
beardatalab = data['traindatalab']
# log_file = open('MBN_bear成功版本.txt', 'w')

# for i in range(10):
#     log.log(beardatalab[4700*i])

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
#     elif(beardatalab[i]!=0 and beardatalab[i]!=1):
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

# flaglab=0
# for img, label in beartrain_loader:
#     if(flaglab<4):
#         for i in range(img.shape[0]):
#             # log.log(img[i])
#             if(flaglab==label[i]):
#                 plt.subplot(10,1,flaglab+1)
#                 plt.title(label[i].numpy())
#                 plt.plot(img[i].numpy())
#                 flaglab+=1
#     else:
#         plt.show()
#         break
# class BranchCNN(nn.Module):
#     def __init__(self):
#         super(BranchCNN,self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(1,8,kernel_size=9),
#             nn.BatchNorm1d(8),
#             nn.ReLU(inplace=True))
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(8,16,kernel_size=16),
#             nn.BatchNorm1d(16),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=4,stride=2))
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(16,32,kernel_size=32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(inplace=True))
#         self.layer4 = nn.Sequential(
#             nn.Conv1d(32,64,kernel_size=64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=4,stride=2))        
#         self.fc = nn.Sequential(
#             nn.Linear(64*445,128),
#             # nn.ReLU(inplace=True),
#             # nn.Linear(1024,64),
#             nn.ReLU(inplace=True),
#             nn.Linear(128,4))       
             
#         self.b1Layer1= nn.Sequential(
#             nn.MaxPool1d(kernel_size=8,stride=2))
#         self.b1Layer2= nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Linear(7944,4),
#             nn.ReLU(inplace=True),
#             nn.Linear(4,4))
#         self.b2Layer1= nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Linear(15792,5),
#             nn.ReLU(inplace=True),
#             nn.Linear(5,5))   
#     def get_label_change(self,label_in1,branch):
#         label_in=label_in1.clone()
#         # log.log(l)  
#         if (branch==1):
#             for i in range(label_in.shape[0]):
#                 if(label_in[i]==7): 
#                     label_in[i]=2
#                 elif(label_in[i]!=0 and label_in[i]!=1):
#                     label_in[i]=3
#         elif(branch==2):
#             for i in range(label_in.shape[0]):
#                 if(label_in[i]==3): 
#                     label_in[i]=0
#                 elif(label_in[i]==5):
#                     label_in[i]=1
#                 elif(label_in[i]==8):
#                     label_in[i]=2
#                 elif(label_in[i]==9):
#                     label_in[i]=3
#                 else:
#                     label_in[i]=4
#         elif(branch==0):
#             for i in range(label_in.shape[0]):
#                 if(label_in[i]==2): 
#                     label_in[i]=0
#                 elif(label_in[i]==4):
#                     label_in[i]=1
#                 elif(label_in[i]==6):
#                     label_in[i]=2
#                 else:
#                     label_in[i]=3
#         return label_in
#     # def branch_crossEntropyLoss(self,out,label,barnnch)
#     def forward(self,x):
#         out1 = self.forward(x,1)
#         if(out1[0].argmax()!=3):
#             if(pre==0):
#                 return 0
#             elif(pre==1):
#                 return 1
#             elif(pre==2):
#                 return 7

        

#     def get_forward_loss(self,x,label_in):
        
#         criterion = nn.CrossEntropyLoss()
#         out1 = self.forward(x,1)
#         out2 = self.forward(x,2)
#         out0 = self.forward(x,0)
#         out=out0[:,0]
#         # log.log(out.shape)
#         for i in range(out1.shape[0]):
#             pre=out1[i].argmax()
#             # log.log(out1[i])
#             # log.log(out1[i].max(1))
            
#             if(pre==0):
#                 out[i]=0
#             elif(pre==1):
#                 out[i]=1
#             elif(pre==2):
#                 out[i]=7
#             pre=out2[i].argmax()
#             if(pre==0):
#                 out[i]=3
#             elif(pre==1):
#                 out[i]=5
#             elif(pre==2):
#                 out[i]=8
#             elif(pre==3):
#                 out[i]=9 
#             pre=out0[i].argmax()
#             if(pre==0):
#                 out[i]=2
#             elif(pre==1):
#                 out[i]=4
#             elif(pre==2):
#                 out[i]=6
            
#             # else:
#             #     out[i]=0

#         # log.log('Counter(data)\n',Counter(label_tmp.cpu().numpy())) # 调用Counter函数
#         # log.log('==========')

#         # log.log(label_tmp.shape)
#         # log.log(out.shape)
#         return out,criterion(out1, self.get_label_change(label_in,1))+\
#             criterion(out2, self.get_label_change(label_in,2))+\
#             criterion(out0, self.get_label_change(label_in,0))
#         # out= F.softmax(out,dim=1)
#         # loss1=0
#         # if(branch==1):        
#         #     for i in range(label_tmp.shape[0]):
#         #         # log.log(out[i][label][i])    
#         #         # loss1+=-torch.log(out[i][label[i]])    
#         #         if(label_tmp[i]!=3): 
#         #             loss1+=-torch.log(out[i][label_tmp[i]])*5.0/3.0
#         #         else:
#         #             loss1+=-torch.log(out[i][label_tmp[i]])*5.0/7.0      
#         #     loss1/=label_tmp.shape[0]
#         # return out,loss1
        

#     def forward(self,x,branch):
#         x = self.layer1(x)
#         if(branch==1):
#             # log.log(label)           
#             x=self.b1Layer1(x)
#             x = x.view(x.size(0),-1)
#             x=self.b1Layer2(x)
#             # x=F.softmax(x,dim=1)
#         else:
#             x=self.layer2(x)
#             if(branch==2):
#                 x= x.view(x.size(0),-1)
#                 x=self.b2Layer1(x)
#             elif(branch==0):
#                 x = self.layer3(x)
#                 x = self.layer4(x)
#                 # log.log(x.shape)   
#                 x = x.view(x.size(0),-1) #第二次卷积的输出拉伸为一行
#                 # log.log(x.shape)       
#                 x = self.fc(x)       
#         return x 
   
# lr = 0.0002 不均衡收敛
lr = 0.0002
momentum = 0.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model =BranchCNN()
# model.to(device)
model.to(device)
branch=5

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
log.log('平安喜乐')
model.load_state_dict(torch.load('MBN_bear成功版本model.pkl'))
model.eval()
eval_acces_class_num=[0]*10
eval_acces_class_num_correct=[0]*10
cntbatch=0
eval_loss = 0   
eval_acc = 0

for img, label in test_loader: 
        # label=model.get_label_change(label,branch)
        # if(cntbatch==5):
        #     break
        
    cntbatch+=1
    img=img.to(device)
    label = label.to(device)
    label = label.long()
    img = img.view(img.size(0),1,2000)
    # img = img.view(img.size(0), -1)
    # t=time.time()
    #out = model.forward(img,branch)
    # log.log(time.time()-t)
    # out2=model.forward(img,branch=2)
    # log.log(label.shape)
    out,loss = model.get_forward_loss(img,label) 
    log.log('batch num: %d/%d, loss %f'%(cntbatch,len(beartestdataset)/test_batch_size,loss))
    # label=model.get_label_change(label,branch)
    # criterion(out, label)
    # eval_losses_class[label]
    # 记录误差
    # log.log(label.shape)
    eval_loss += loss.item()
    # 记录准确率
    # _, pred = out.max(1)
    pred=out
    num_correct = (pred == label).sum().item()
    acc = num_correct / img.shape[0]
    eval_acc += acc
    # log.log('Counter(label)\n',Counter(label.cpu().numpy())) # 调用Counter函数
    # log.log('==========')
    for i in range(label.shape[0]):
        # log.log(label.shape[0])
        eval_acces_class_num[label[i]]+=1
        if(pred[i]==label[i]):
            eval_acces_class_num_correct[label[i]]+=1
        else:
            eval_acces_class_num[int(pred[i])]+=1
        # eval_acces_class+=acc_class
eval_acces_class=np.array(eval_acces_class_num_correct)/np.array(eval_acces_class_num)
eval_losses.append(eval_loss / len(test_loader))
eval_acces.append(eval_acc / len(test_loader))
log.log('Test Loss: {:.4f}, Test Acc: {:.4f}'
        .format(eval_loss / len(test_loader), eval_acc / len(test_loader)))
#np.sort(eval_acces_class)
for i in range(10):
    maxsite=np.argmax(eval_acces_class)
    log.log('class {},acc: {:.4f}'.format(maxsite,eval_acces_class[maxsite]))
    eval_acces_class[maxsite]=-1
    # log.log('modle save')
    # torch.save(model.state_dict(), 'MBN_bear成功版本model.pkl')
    # for i in range(10):
    #     log.log('class {},acc: {:.4f}'.format(i,eval_acces_class[i]))

# torch.save(model.state_dict(), 'MBN_bear成功版本model.pkl')    
# for epoch in range(num_epoches):
#     train_loss = 0
#     train_acc = 0
#     model.train()
#     #动态修改参数学习率
#     if epoch%5==0:
#         optimizer.param_groups[0]['lr']*=0.9
#         log.log(optimizer.param_groups[0]['lr'])
       
#     cntbatch=0 
#     for img, label in train_loader:
#         # log.log('Counter(label)\n',Counter(label.cpu().numpy())) # 调用Counter函数
#         # log.log('==========')
#         cntbatch+=1
     
#         img=img.to(device)
#         label = label.to(device)
#         label=label.long()
#         # img = img.view(img.size(0), -1)
#         # 前向传播
#         img = img.view(img.size(0),1,2000)

#         out,loss=model.get_forward_loss(img,label)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # 记录误差
#         train_loss += loss.item()
#         # 保存loss的数据与epoch数值
#         writer.add_scalar('Train', train_loss/len(train_loader), epoch)
#         # 计算分类的准确率
#         # _, pred = out.max(1)
#         pred=out
#         # log.log('Counter(label)\n',Counter(label.cpu().numpy())) # 调用Counter函数
#         # log.log('==========')
#         num_correct = (pred == label).sum().item()
#         acc = num_correct / img.shape[0]
#         train_acc += acc
#         log.log('epoch: %d/%d,batch num: %d/%d, loss %f'%(epoch,num_epoches,cntbatch,len(beartraindataset)/train_batch_size,loss))
#         # if(cntbatch==5):
#         #     break
        
#     losses.append(train_loss / len(train_loader))
#     acces.append(train_acc / len(train_loader))
#     # 在测试集上检验效果
#     eval_loss = 0   
#     eval_acc = 0
#     #net.eval() # 将模型改为预测模式
#     model.eval()
#     if(branch==1 or branch==0):
#         eval_acces_class_num=[0]*4
#         eval_acces_class_num_correct=[0]*4
#     elif branch==2:
#         eval_acces_class_num=[0]*5
#         eval_acces_class_num_correct=[0]*5
#         cntbatch=0
#     else:
#         eval_acces_class_num=[0]*10
#         eval_acces_class_num_correct=[0]*10
#         cntbatch=0
#     for img, label in test_loader: 
#         # label=model.get_label_change(label,branch)
#         # if(cntbatch==5):
#         #     break
        
#         cntbatch+=1
#         img=img.to(device)
#         label = label.to(device)
#         label = label.long()
#         img = img.view(img.size(0),1,2000)
#         # img = img.view(img.size(0), -1)
#         # t=time.time()
#         #out = model.forward(img,branch)
#         # log.log(time.time()-t)
#         # out2=model.forward(img,branch=2)
#         # log.log(label.shape)
#         out,loss = model.get_forward_loss(img,label) 
#         log.log('batch num: %d/%d, loss %f'%(cntbatch,len(beartestdataset)/test_batch_size,loss))
#         # label=model.get_label_change(label,branch)
#         # criterion(out, label)
#        # eval_losses_class[label]
#         # 记录误差
#         # log.log(label.shape)
#         eval_loss += loss.item()
#         # 记录准确率
#         # _, pred = out.max(1)
#         pred=out
#         num_correct = (pred == label).sum().item()
#         acc = num_correct / img.shape[0]
#         eval_acc += acc
#         # log.log('Counter(label)\n',Counter(label.cpu().numpy())) # 调用Counter函数
#         # log.log('==========')
#         for i in range(label.shape[0]):
#             # log.log(label.shape[0])
#             eval_acces_class_num[label[i]]+=1
#             if(pred[i]==label[i]):
#                 eval_acces_class_num_correct[label[i]]+=1
#             else:
#                 eval_acces_class_num[int(pred[i])]+=1
#         #eval_acces_class+=acc_class

        
#     eval_acces_class=np.array(eval_acces_class_num_correct)/np.array(eval_acces_class_num)

#     eval_losses.append(eval_loss / len(test_loader))
#     eval_acces.append(eval_acc / len(test_loader))
#     log.log('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
#           .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader), 
#                      eval_loss / len(test_loader), eval_acc / len(test_loader)))
#     #np.sort(eval_acces_class)
#     for i in range(10):
#         maxsite=np.argmax(eval_acces_class)
#         log.log('class {},acc: {:.4f}'.format(maxsite,eval_acces_class[maxsite]))
#         eval_acces_class[maxsite]=-1
#     log.log('modle save')
#     torch.save(model.state_dict(), 'MBN_bear成功版本model.pkl')
#     # for i in range(10):
#     #     log.log('class {},acc: {:.4f}'.format(i,eval_acces_class[i]))
    

