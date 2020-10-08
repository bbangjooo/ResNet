import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn,cuda,optim
from torch.autograd import Variable
import time
# Setting

device= 'cuda' if cuda.is_available()  else 'cpu'
download_root="cifar10/"
batch_size=128
learning_rate=0.1
momentum=0.9            # ????
weight_decay = 0.0001   # ????
transforms=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Dataset

train_set=CIFAR10(download_root,train=True,transform=transforms,download=True)
test_set=CIFAR10(download_root,train=False,transform=transforms)

# Dataloader

train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)

# Model

# block => 3x3conv-bn-relu-3x3conv-addition-relu

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class Residual_Block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,downsample=False):
        super(Residual_Block,self).__init__()
        self.downsample=downsample
        # 3x3conv-bn-relu-3x3conv-addition-relu
        self.before_add=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1)
        )
        self.down_Sampling=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channel)
        )
        self.ReLU=nn.ReLU(inplace=True)

    def forward(self,x):
        shortcut = x
        output=self.before_add(x)
        if self.downsample==True:            
            shortcut=self.down_Sampling(shortcut)
        #print ("[*] Output")
        #print (output.shape)
        #print ("[*] ShortCut")
        #print (shortcut.shape)
        output+=shortcut
        output=self.ReLU(output)
        return output

class ResNet_CIFAR10(nn.Module):
    def __init__(self,n):
        super(ResNet_CIFAR10,self).__init__()
        self.n=2*n
        self.conv1=nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1)
        self.layer1=[Residual_Block(16,16,stride=1)]
        for _ in range(self.n-1):
            self.layer1.append(Residual_Block(16,16,1))
        self.layer2=[Residual_Block(16,32,stride=2,downsample=True)]
        for _ in range(self.n-1):
            self.layer2.append(Residual_Block(32,32,1))
        self.layer3=[Residual_Block(32,64,stride=2,downsample=True)]
        for _ in range(self.n-1):
            self.layer3.append(Residual_Block(64,64,1))

        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Sequential(*self.layer2)
        self.layer3 = nn.Sequential(*self.layer3)

        self.avgpool=nn.AvgPool2d(kernel_size=8)
        self.fc=nn.Linear(64,10)
        self.apply(_weights_init)

    def forward(self,x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

model=ResNet_CIFAR10(3)
model.to(device)
# Loss & Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)

# Train

def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=Variable(data),Variable(target)
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



# Test

def test():
    model.eval()
    loss=0
    correct=0
    for data,target in test_loader:
        data,target=Variable(data),Variable(target)
        data,target=data.to(device),target.to(device)
        output=model(data)
        loss+=nn.functional.cross_entropy(output,target,reduction='sum').item()
        pred=output.data.max(1,keepdim=True)[1]
        correct+=pred.eq(target.data.view_as(pred)).cpu().sum()
    loss/=len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, TOP 1 ERROR: {}/{} ({:.0f}%)\n'.format(
        loss, len(test_loader.dataset)-correct, len(test_loader.dataset),
        100. * (len(test_loader.dataset)-correct) / len(test_loader.dataset)))


if __name__ == "__main__":
    start=time.time()
    for epoch in range(20):
        train(epoch)
        test()
    print ("[*]  total elapsed time : ",time.time()-start)