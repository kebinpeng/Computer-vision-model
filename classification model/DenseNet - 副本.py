#encoding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Denseblock(nn.Module):#这个是密集连结层
    def __init__(self,inDem,growth_rate):#growth_rate是每个卷积层输出的通道数，inDem代表输入的维度
        super(Denseblock,self).__init__()
        self.bn1 = nn.BatchNorm2d(inDem)
        self.conv1 = nn.Conv2d(inDem, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
    def forward(self,x):
        y = self.bn1(x)
        y = F.relu(y)
        y = self.conv1(y)


        y = self.bn2(y)
        y = F.relu(y)
        y = self.conv2(y)
        y = torch.cat([y,x],1)
        return y

class Transition(nn.Module):#这个层是用来调整输入输出大小的，他会把输入的长和宽减半
    def __init__(self,inDem,outDem):
        super(Transition,self).__init__()
        self.bn = nn.BatchNorm2d(inDem)
        self.conv = nn.Conv2d(inDem,outDem,kernel_size=1,bias=False)
    def forward(self,x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = F.avg_pool2d(x,2)
        return x

class DenseNet(nn.Module):
    def __init__(self,block,nblock,growth_rate=12, reduction=0.5, num_classes=10):#block代表模块的名称(是dense模块还是Transition)
                                                                                  #nblock代表有多少dense模块，每个模块几层
        super(DenseNet,self).__init__()
        self.growth_rate = growth_rate

        num = 2*growth_rate
        self.conv1 = nn.Conv2d(3,num,kernel_size=3,padding=1,bias=False)

        self.dense1 = self.make_dense_layer(block,num_classes,nblock[0])
        out_classes = int(math.floor(num_classes*reduction))
        self.trans1 = Transition(num_classes, out_classes)
        num_classes = out_classes

        self.dense2 = self.make_dense_layer(block, num_classes, nblock[1])
        num_classes += nblock[1]*growth_rate
        out_classes = int(math.floor(num_classes*reduction))

        self.trans2 = Transition(num_classes, out_classes)
        num_classes = out_classes

        self.dense3 = self.make_dense_layer(block, num_classes, nblock[2])
        num_classes += nblock[2]*growth_rate
        out_planes = int(math.floor(num_classes*reduction))
        self.trans3 = Transition(num_classes, out_classes)
        num_classes = out_classes
        
        self.dense4 = self.make_dense_layer(block, num_classes, nblock[3])
        num_classes += nblock[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_classes)
        self.linear = nn.Linear(num_classes, num_classes)

    def make_dense_layer(self,block,inDem,nblock):
        layer = []
        for i in range(nblock):
            layer.append(block(inDem,self.growth_rate))
            inDem += self.growth_rate
        return nn.Sequential(*layer)

    def forward(self,x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)

       
        x = self.dense2(x)
        x = self.trans2(x)

        x = self.dense3(x)
        x = self.trans3(x)
        
        x = self.dense4(x)

        x = self.bn(x)
        x = F.relu(x,4)
        x = F.avg_pool2d(x)
        x=x.view(x.size(0),-1)
        x = self.linear(x)

        return x


if __name__ == '__main__':
    net = DenseNet(Denseblock,[6,12,24,16],growth_rate=12)
    #print(net)

    x = torch.randn(1,3,32,32)
   # print(type(x))
    y = net(x)

    print(y)




