import os
import torch.nn.functional as F


def create_modules(cfgfile_path):
	"""
    Purpose:
        Create modules based on layers in cfg file 
    Args:
        cfgfile_path: type-str
    Return:
        net_info, nn.ModuleList(nn.Sequential)
    Process:
        1. process cfgfile and get module layer info
        2. create module based on each layer and store them into nn.Sequential
        3. add nn.Sequential to nn.ModuleList
        4. return 
	"""
    pass

class EmptyLayer(nn.Module):
    """
    Placeholder for 'route' and 'shortcut' layers
    """
    pass

class Upsample(nn.Module):
    """
    Purpose:
        make the dimention enlarge and suitable to concanate or do some operation
    Custom Upsample layer (nn.Upsample gives deprecated warning message)
    should use nn.interpolate , the same parameters as nn.Upsample
    """
    pass

class YOLOLayer(nn.Module):
    """
    Purpose:
        rearrange the data from former layer which has the form :
            [batch, 255, grid_size, grid_size]
        if the model in training status, then return:
            [batch,3,grid_size, grid_size, 85]
        if in inferrence, return:
            [batch,-1, 85], put all the anchors in the second dimention

    return the result based on the model.training status
    Process:
        1.

    """
    pass

class Darknet(nn.Module):
    """
    Purpose:
        YOLOv3 detection model
        process images parameter
    Process:
        1. process the x using different approach for different layers
        2. Conv -> process x directly
        3. shortcut -> same as residue addtion: add former layer and 'from' layer
        4. yolo -> precess x directly
        5. route -> connect layers in the route
        6. upsample -> precess x directly
    """
    pass

def get_yolo_layers(model):
    """
    Purpose:
        finde the yolo layer index in the model
    Return:
        list(int)
    """
    pass

def create_grids(self, img_size, ng, device='cpu'):
    """
    Purpose:
        create model parameters, which can be used directly in the model.
    Args:
        self : refers to model 
        img_size: int, input image size
        ng: tuple of two ,(ny, nx), x and y grid size
    """
    pass

def load_darknet_weights(self, weights,cutoff = -1):
    """
    Purpose:
        parse and loads the weights stored in 'weights'
    Args:
        self : model
        weights: file path ends with .weight
        cutoff: save layers between 0 and cutoff (if cutoff=-1 all are saved)
    Return:
        cutoff
    Process:
        1. first five data of weight are header values

    """
    pass

def save_weights(self, path='model.weights', cutoff=-1):
    """
    Purpose:
        convert a pytorch model to Darknet format (*.pt to *.weights)

    """
    pass

def convert(cfg='yolov3.cfg', weights='weights/yolov3.weights'):
    """
    Purpose:
        Converts between Pytorch and Darknet format per extension
        i.e. *.weights -> *.pt and vice versa

    """
    pass



