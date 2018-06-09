#encoding = utf-8
from torch import nn
from torch.nn import functional as F

class ResBlock(nn.Module):#残差块
    def __init__(self,InDemention,outDemention,stride=1,shortcut=None):
        super(ResBlock, self).__init__()
        self.main= nn.Sequential( #残差块中间的主要结构 
                nn.Conv2d(InDemention,outDemention,3,stride,1,bias=False),
                nn.BatchNorm2d(outDemention),
                nn.ReLU(inplace=True),
                nn.Conv2d(outDemention,outDemention,3,1,1,bias=False),
                nn.BatchNorm2d(outDemention),

                )   
        self.right = shortcut#残差块右边的结构
        
    def forward(self,x):
        x = self.main(x)
        if shortcut == None:
            res = self.right 
        else:
            res = self.right(x)

        return F.relu(out)

class ResNet34(nn.Module):

    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'

        self.first = nn.Sequential(
                nn.Conv2d(3,64,7,2,3,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3,2,1)#提取特征
         )

        self.layer1 = self.make_res_layer(64,128,3)

        self.layer2 = self.make_res_layer(128,256,4,stride=2)

        self.layer3 = self.make_res_layer(256,512,6,stride=2)
        self.layer4 = self.make_res_layer(512,512,3,stride=2)

        self.fc = nn.Linear(512,num_classes)#全连接层


    def make_res_layer(self,InDemention,outDemention,block_num,stride=1):
         shortcut = nn.Sequential(
                 nn.Conv2d(InDemention,outDemention,1,stride,bias= False),
                 nn.BatchNorm2d(outDemention),
         )

         layer = []
         layer.append(ResBlock(InDemention,outDemention,stride,shortcut))
         
         for i in range(1,block_num):
             layer.append(ResBlock(InDemention,outDemention))
         
         return nn.Sequential(*layer)
     
    def forward(self,x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    net = ResNet34()
    print(net)

     


