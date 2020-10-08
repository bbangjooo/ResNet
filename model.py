import torch
from torch import nn

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