import argparse
import time
from sys import platform
import torch
import numpy as np 
from torch.utils.data import DataLoader, Dataset

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        names, # class names
        imgs_path, # original test image path
        weights,
        batch_size,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        debug = False,

):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model

    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    dataset = Image_dataset_detect(imgs_path,img_size,debug = debug)

    dataloader = DataLoader(dataset, batch_size= batch_size)

    # Get classes and colors
    classes = load_classes(names)
    num_classes = len(classes)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    print('Finished loading data')
    
    # get the predicted bboxes
    print('Predicting bboxes...')
    ratio_list = []
    bboxes_list = []
    img_path_list = []
    
    for batch in dataloader:
        resized_images, ratio, img_path = batch
        img_path_list += list(img_path)
        ratio_list += ratio
        predictions,_ = model(resized_images)
        bboxes = write_results(predictions, confidence=conf_thres,num_classes=num_classes,nms_conf=nms_thres)

        bboxes_list += bboxes

    print('Drawing bboxes...')
    object_detect(img_path_list,img_size,classes, bboxes_list,ratio_list,colors,output_path = output)
    print('\n Done \n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='coco.names file path')
    parser.add_argument('--imgs_path', type=str, help='file path contain images')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of DataLoader')
    parser.add_argument('--output', type=str, default='output', help='marked images to store')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--debug', type=bool, default=False, help='use debug mode to test the script')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            cfg =opt.cfg,
            names = opt.names,
            imgs_path= opt.imgs_path,
            weights = opt.weights,
            batch_size = opt.batch_size,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            debug = opt.debug,
        )
