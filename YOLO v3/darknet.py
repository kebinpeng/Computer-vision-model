#encoding=utf-8
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from parse_cfg import *
from util import *
import cv2

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):#这个是造模块的函数，把解析好的配置文件读进来，然后生成各个模块，等于是把网络的__init__木块拆解出来
    net=blocks[0]#配置文件第0项是对整个网络配置参数
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index,x in enumerate(blocks[1:]):#配置文件第一项往后，就是对网络的各种模块进行定义
        module = nn.Sequential()
#        print('index is ',end="")
#        print(index)
#        print('x is',end="")
#        print (x)可以打印出来看看每个参数是啥样子
        if (x["type"] == "convolutional"):
           
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

           
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

           
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

      
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        elif (x["type"] == "route"):#route layer就是把前头若干层连接在一起
            x["layers"] = x["layers"].split(',')#yolov3会在第25层把16层链接起来
                                                #第27层把26和24链接起来
            start = int(x["layers"][0])
            
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        
        elif x["type"] == "shortcut":#shortcut就是捷径的意思，类似resnet，从某些层跳走了
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
       
    return (net, module_list)



class Darknet(nn.Module):#这个是网络的主体结构,实现向前传播的过程
     def __init__(self, cfgfile):#__init__的主要功能在create_modules函数已经实现了
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
     def forward(self,x,CUDA):
         module = self.blocks[1:]
         output = {}
         write=0
         for i,module in enumerate(module):
             module_type = (module["type"])

             if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
    
             elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = output[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = output[i + layers[0]]
                    map2 = output[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
    
             elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = output[i-1] + output[i+from_]
    
             elif module_type == 'yolo':        
                anchors = self.module_list[i][0].anchors
                
                inp_dim = int (self.net_info["height"])
        
                
                num_classes = int (module["classes"])
        
                
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:              
                    detections = x
                    write = 1
        
                else:       
                    detections = torch.cat((detections, x), 1)
        
             output[i] = x
        
         return detections



if __name__== '__main__':
    cfgfile="C:\\Users\\pkb\\Desktop\\computer vision\\pytorch\\YOLO v3\\cfg\\yolov3.cfg"
    #blocks=parse_cfg(cfgfile)
    print( create_modules(blocks))

    picture = "C:\\Users\\pkb\\Desktop\\computer vision\\pytorch\\YOLO v3\\person.jpg"
    img = cv2.imread(picture)
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img0 =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img0 = img0[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img0 = torch.from_numpy(img0).float()     #Convert to float
    img0 = Variable(img0)                     # Convert to Variable
   
    model = Darknet(cfgfile)
    inp = img0
    pred = model(inp, torch.cuda.is_available())
    print (pred[0].size())
    print (pred[0][0])
    #print (pred.output_filters)

