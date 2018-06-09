#encoding=utf-8
def parse_cfg(cfgfile):#解析cfg配置文件
    
    #cfgfile是指配置文件的地址
    #cfgfile="C:\Users\pkb\Desktop\工作\pytorch\YOLO v3\cfg\yolov3.cfg"
    file = open(cfgfile, 'r')#把空行和注释全去掉
    lines = file.read().split('\n')                        
    lines = [x for x in lines if len(x) > 0]                
    lines = [x for x in lines if x[0] != '#']              
    lines = [x.rstrip().lstrip() for x in lines]           
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[": #新模块以[开始              
            if len(block) != 0:          
                blocks.append(block)     
                block = {}               
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks
