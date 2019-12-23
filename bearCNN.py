import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import datasets,transforms
import scipy.io as scio
import numpy as np
train_batch_size = 50
test_batch_size=50
learning_rate = 0.02
num_eporches =10

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
 
beartrain_loader = DataLoader(beartraindataset, batch_size=train_batch_size, shuffle=True)
beartest_loader = DataLoader(beartestdataset, batch_size=test_batch_size, shuffle=False)
train_loader=beartrain_loader
test_loader = beartest_loader

# CNN模型
# 建立四个卷积层网络 、两个池化层 、 1个全连接层
# 第一层网络中，为卷积层，将28*28*1的图片，转换成16*26*26
# 第二层网络中，包含卷积层和池化层。将16*26*26 -> 32*24*24,并且池化成cheng32*12*12
# 第三层网络中，为卷积层，将32*12*12 -> 64*10*10
# 第四层网络中，为卷积层和池化层，将 64*10*10 -> 128*8*8,并且池化成128*4*4
# 第五次网络为全连接网络

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





model = CNN()
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=learning_rate)

for i in range(num_eporches):
    epoch = 0
    for data in train_loader:
        img, label = data
    
        # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵
        #img = img.view(img.size(0), -1)
        
        if torch.cuda.is_available():
            img = img.cuda()            
            label = label.long().cuda()
            # layer=nn.Sequential(
            #     nn.Conv1d(1,16,kernel_size=9),
            #     # nn.Conv2d(1,16,kernel_size=3),
            #     nn.BatchNorm1d(16),      
            #     nn.ReLU(inplace=True)).cuda()  
        else:
            img = Variable(img)        
            label = Variable(label).long()

        # print(img.size(0))
        img = img.view(img.size(0),1,2000)   
        # img = img.permute(0, 2, 1)
        # conv1 = nn.Conv1d(in_channels=1,out_channels = 16, kernel_size = 9)
        # input = torch.randn(32, 35, 256)
        # # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
        # input = input.permute(0, 2, 1)
        # input = Variable(input)
        # out = conv1(input)
        # print(out.size())



        # print(img.shape)
        
        # print(layer(img).shape)
        


        out = model(img)
        loss = criterion(out, label)
        print_loss = loss.data.item()
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch+=1
        if epoch%50 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
            model.eval()
            eval_loss = 0
            eval_acc = 0
            for data in test_loader:
                img, label = data
                
                # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵
                #img = img.view(img.size(0), -1)
                
                if torch.cuda.is_available():
                    img = img.cuda()
                    label = label.cuda().long()
                img = img.view(img.size(0),1,2000)  
                out = model(img)
                loss = criterion(out, label)
                eval_loss += loss.data.item()*label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(
                eval_loss / (len(beartestdataset)),
                eval_acc / (len(beartestdataset))
            ))

