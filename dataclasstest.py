#测试MNIST数据集，不同类别样本的分类难易程度
import numpy as np
import torch
# 导入 pytorch 内置的 mnist 数据
from torchvision.datasets import mnist 
#import torchvision
#导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
#导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import scipy.io as scio
# 定义一些超参数
train_batch_size = 200
test_batch_size = 100
num_epoches = 80

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

#定义预处理函数
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
#下载数据，并对数据进行预处理
train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=True)
test_dataset = mnist.MNIST('./data', train=False, transform=transform)

path = 'matlab.mat'
data = scio.loadmat(path)
beardata=data['traindata'] 
beardatalab = data['traindatalab']

index = [i for i in range(len(beardata))] 
np.random.shuffle(index)
beardata = beardata[index]
beardatalabtmp = beardatalab[index]

MAX=(np.amax(beardata))
MIN=(np.amin(beardata))
beardatalab=np.zeros(47000)
for i in range(47000):
    beardatalab[i]=beardatalabtmp[i][0]

beardata=(((beardata-MIN)/(MAX-MIN)-0.5)*2).astype(np.float32)

# beardatalab.dtype="float32"

beartraindata=beardata[0:40000]
beartraindatalab=beardatalab[0:40000]
beartestdata=beardata[40000:47000]
beartestdatalab=beardatalab[40000:47000]

# print(beartraindata.shape)
# print(beartraindatalab.shape)
# print(type(beardata))
# print(beartestdata.shape)
# print(beartestdatalab.shape)
#得到一个生成器

beartraindataset=TensorDataset(torch.from_numpy(beartraindata),torch.from_numpy(beartraindatalab))
beartestdataset=TensorDataset(torch.from_numpy(beartestdata),torch.from_numpy(beartestdatalab))

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

beartrain_loader = DataLoader(beartraindataset, batch_size=train_batch_size, shuffle=True)
beartest_loader = DataLoader(beartestdataset, batch_size=test_batch_size, shuffle=False)
train_loader=beartrain_loader
test_loader = beartest_loader
# plt.figure(1)
# flaglab=0
# for img, label in beartrain_loader:
#     print(type(label.shape))
#     print(label.shape)   
#     print(img)
    
    
# for img, label in train_loader:
#     print(img)
#     print(type(label.shape))
#     print(label.shape)   
    
# b0layer1 = nn.Linear(784, 100) 
# for img, label in train_loader:
#     img = img.view(img.size(0), -1)
#     print((img[0])) 
#     print(type(img))
#     print(img.shape)
#     # print(label.shape)   
#     # print(b0layer1(img[0])) 
#     break
# layer1 = nn.Linear(2000, 100) 
flaglab=0
# for img, label in beartrain_loader:
    # img = img.view(img.size(0), -1)
    # print((img[0])) 
    # print(type(img))
    # print(img.shape)
    # # print(label.shape)   
    # print(layer1(torch.tensor(img[0], dtype=torch.float32))) 
    # break
    # if(flaglab<10):
    #     for i in range(img.shape[0]):
    #         # print(img[i])
    #         if(flaglab==label[i]):
    #             plt.subplot(10,1,flaglab+1)
    #             plt.title(label[i].numpy())
    #             plt.plot(img[i].numpy())
    #             flaglab+=1
    # else:
    #     plt.show()
    #     break
    

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=9),
            # nn.Conv2d(1,16,kernel_size=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16,32,kernel_size=16),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4,stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(32,64,kernel_size=32),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv1d(64,128,kernel_size=64),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4,stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(128*445,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10))
    
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)   
        x = x.view(x.size(0),-1) #第二次卷积的输出拉伸为一行
        # print(x.shape)       
        x = self.fc(x)
        return x


class Net(nn.Module):
    """
    使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
    """
    def __init__(self):
        super(Net, self).__init__()
        in_dim=2000
        n0_hidden_1=1000
        n1_hidden_2=100
        n1_hidden_3=50
        out1_dim=10
        n2_hidden_2=20
        out2_dim=10
        self.b0layer1 = nn.Linear(in_dim, n0_hidden_1) 
        self.b1layer2 = nn.Linear(n0_hidden_1, n1_hidden_2)
        self.b1layer3 = nn.Linear(n1_hidden_2, n1_hidden_3)
        self.b1layer4 = nn.Linear(n1_hidden_3, out1_dim)

        # self.b2layer2 = nn.Sequential(nn.Linear(n0_hidden_1, n2_hidden_2))
        # self.b2layer3 = nn.Sequential(nn.Linear(n2_hidden_2, out2_dim))
        self.b2layer2 = nn.Linear(n0_hidden_1, out2_dim)
  
    def forward(self, x, branch):
        # print(x)
        x=F.relu(self.b0layer1(x))
        if(branch==1):
            x = F.relu(self.b1layer2(x))
            x = F.relu(self.b1layer3(x))
            x = self.b1layer4(x)            
        elif(branch==2):
            # x = F.relu(self.b2layer2(x))
            x = self.b2layer2(x)     
            # x=F.softmax(x,dim=1)
        return x#F.softmax(x,dim=1)    
lr = 0.8
momentum = 0.9
#实例化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs")
#    # dim = 0 [20, xxx] -> [10, ...], [10, ...] on 2GPUs
#    model = nn.DataParallel(model)
model =Net()
model.to(device)
branch=2
# 定义损失函数和优化器
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
        img = img.view(img.size(0), -1)
        # 前向传播
        out = model.forward(img,branch)
        #out2 = model.forward(img,branch=2)
        loss = criterion(out, label)#+criterion(out2, label)+1000000
        loss1=0
        out= F.softmax(out,dim=1)
        for i in range(label.shape[0]):
            # print(out[i][label][i])           
            loss1+=-torch.log(out[i][label[i]])
        loss1/=label.shape[0]
        # print(out.shape)
        # print(loss1.shape)
        # print(loss)
        # print(loss1)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # loss1.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 保存loss的数据与epoch数值
        writer.add_scalar('Train', train_loss/len(train_loader), epoch)
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    # 在测试集上检验效果
    eval_loss = 0   
    eval_acc = 0
    #net.eval() # 将模型改为预测模式
    model.eval()
    
    eval_acces_class_num=[0]*10
    eval_acces_class_num_correct=[0]*10
    for img, label in test_loader: 
        img=img.to(device)
        label = label.to(device)
        label = label.long()
        img = img.view(img.size(0), -1)
        out = model.forward(img,branch)
        
        # out2=model.forward(img,branch=2)
        loss = criterion(out, label)
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
    for i in range(10):
        maxsite=np.argmax(eval_acces_class)
        print('class {},acc: {:.4f}'.format(maxsite,eval_acces_class[maxsite]))
        eval_acces_class[maxsite]=-1
        
    # for i in range(10):
    #     print('class {},acc: {:.4f}'.format(i,eval_acces_class[i]))
    

