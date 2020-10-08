import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn,cuda,optim
from torch.autograd import Variable
import time
from model import *
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