from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *

def get_test_input():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img,(608,608))
    img_ = img[:,:,::-1].transpose((2,0,1)) # BGR -> RGB | H*W*C -> C*H*W
    img_ = img_[np.newaxis,:,:,:] / 255.0 # add a channel at 0 (for batch) | normalise
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_) # convert to variable
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    

    """
    file = open(cfgfile,'r')
    lines = file.read().split('\n') 
    lines = [x for x in lines if len(x) >0] # 去除空行
    lines = [x for x in lines if x[0] != '#'] #去除注释行
    lines = [x.strip() for x in lines] 

    block = {} # store as dictionary
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block) # 把上一个block添加到blocks中
                block= {}
            block['type'] = line[1:-1].strip() # 下一个block的类型名称
        else:
            key,value = line.split('=')
            block[key.strip()] = value.strip()

    blocks.append(block) # 把最后一个block 也加入

    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3 # the image has three channels
    output_filters = []

    for index, x in enumerate(blocks[1:]): # blocks starts from 1 because the first one is net info
        module = nn.Sequential()
        if x['type'] == 'convolutional': # CONV -> BN -> activation
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size -1)//2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size,stride,pad,bias = bias)
            module.add_module('conv_{0}'.format(index), conv)

            # add the batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            # check the activation
            if activation == 'leaky':
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module('leaky_{0}'.format(index), activn)

        # upsample layer
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor = 2, mode= 'bilinear', align_corners=True)
            module.add_module('upsample_{}'.format(index), upsample)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            # start of a route
            start = int(x['layers'][0])
            #end, if there exists one.
            try:
                end = int(x['layers'][1])
            except:
                end = 0

            if start > 0:
                start = start - index # the index is the largest number layer now
            if end > 0: # to ensure that the end is smaller than the index
                end = end - index

            route = EmptyLayer()
            module.add_module('route_{}'.format(index),route)
            if end < 0 : # 目的为保证end and start都是负数，保证后面的统一性
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format(index), shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask] # 要使用第几个anchors

            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)] # 每个anchors有两个数，分别代表h,w
            anchors = [anchors[i] for i in mask] # 选择所需要的anchors

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)


    return(net_info, module_list)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


#blocks = parse_cfg('./cfg/yolov3.cfg')
#print(create_modules(blocks))

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self,x, CUDA):
        detection = [] # 对yolo的结果进行储存
        modules = self.blocks[1:]
        outputs = {} # 储存每一次moudle的结果
        write = 0

        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)

            elif module_type == 'route': # 直接提取之前的feature map 然后在进行拼接（num>2）,增加filter数目
                layers = module['layers']
                layers = [int(a) for a in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] -i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] >0:
                        layers[1] = layers[1] -i

                    map1 = outputs[i+layers[0]]
                    map2 = outputs[i+ layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut': #  shortchu 对残积进行拼接，前一层和from的那一层直接是相同位置的数直接相加
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors

                inp_dim = int(self.net_info['height'])

                num_classes = int(module['classes'])

                # transform
                # 在到yolo层时，从cfg文件中可以看出filter 层数为225（3*85），3个anchors，85个attributes
                #所以现在 x 的shape 为[Batch,225,N,N] , N 为grid cell 的个数
                # predict_transform 的作用是把[Batch,225,N,N] -> [Batch,3,N,N,85]
                x = x.data
                x = predict_transform(x, inp_dim, anchors,num_classes,CUDA)

                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x # outputs 储存每一次module处理的结果，以供后面进行shortcut和route操作

        try:
            return detections
        except:
            return 0

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype = np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()

                    # load weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    # cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)


                else:
                    #conv biases
                    num_biases = conv.bias.numel()

                    # load the weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr += num_biases

                    # reshape the loaded weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # copy the data
                    conv.bias.data.copy_(conv_biases)


                #load the conv weights
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


#model = Darknet('cfg/yolov3.cfg')
#model.load_weights('../pytorch-yolo-v3/yolov3.weights')
#inp = get_test_input()
#pred = model(inp, torch.cuda.is_available())
#print(pred, pred.shape)



