#coding:utf-8


from torch import nn
from torch.nn import functional as F

class vgg16(nn.Module):
     def __init__(self,  num_classes=1000,init_weights=True):
        super(vgg16, self).__init__()
        self.model_name = 'vgg16'
        
        self.feature = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,64,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64,128,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128,128,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128,256,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,256,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256,256,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256,512,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512,512,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,512,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
         )
        self.classifier = nn.Sequential(
                nn.Linear(512*7*7,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096,num_classes),
         )

        if init_weights:
            self.init()
        
     def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)

     def init(self): 
         for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    net = vgg16()
    print(net)




