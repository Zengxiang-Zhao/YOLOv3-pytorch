from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
from torch.utils.data import Dataset,DataLoader
from darknet import Darknet
import pickle as pkl
import random
import copy

def arg_parse():
    """
    Parse arguments to the detect module
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument('--images', dest='images',help="Image/ dictionary containing images", default="imgs",type=str)
    parser.add_argument('--det',dest="det",help='Image / dictionary to store the detections to', default = 'det', type=str)
    parser.add_argument('--bs', dest='bs',  help="Batch size", type=int,default=1)
    parser.add_argument('--confidence', dest='confidence',help='Object confidence to filter predictions', default=0.5, type=float)
    parser.add_argument('--nms_thresh', dest='nms_thresh',help="NMS Threshhold", default=0.4, type=float)
    parser.add_argument('--cfg',dest = 'cfgfile', help='Config file', 
                       default = 'cfg/yolov3.cfg', type=str)
    parser.add_argument('--weights',dest='weightsfile',help='weightsfile',
                        default='yolov3.weights', type=str)
    parser.add_argument('--reso', dest='reso',help=' Input resolution of the network',
                        default = '416',type=str,)
    parser.add_argument('--exame', dest='exame', help='use a sample to check the pipline', default=False, type=bool)

    return parser.parse_args()

# reform the images to match the model's dimention
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim,inp_dim
    ratio = min(w/img_w, h/img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim, inp_dim, 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    canvas = canvas[:,:,::-1].transpose((2,0,1)).copy()
    canvas = torch.from_numpy(canvas).float().div(255.0)
    return canvas, ratio

# before draw bboxes transform the bboxe to make the bbox suit for the original image
def change_bboxes(bbox_predict, ratio, img_w,img_h):
    """
    bbox,ratio,img_w,img_h all numpy values
    return the bboxes based on the original image
    """
    bbox = copy.deepcopy(bbox_predict)
    
    bbox[:,[0,2]] -= (inp_dim - ratio*(img_w))/2.0
    bbox[:,[1,3]] -= (inp_dim -ratio*(img_h))/2.0
    bbox[:,:4] = bbox[:,:4]/ ratio
  
    return bbox


class Image_dataset_test(Dataset):
    def __init__(self,img_list,resize_function,inp_dim):
        super(Image_dataset_test,self).__init__()
        self.img_list = img_list
        self.resize_function = resize_function
        self.inp_dim = inp_dim
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,indx):
        # get image
        img_path = self.img_list[indx]
        img = cv2.imread(img_path)
        
        resized_image, ratio = self.resize_function(img,self.inp_dim)
#         resized_image = resized_image[:,:,::-1]
        return resized_image,ratio,img_path

# draw bboxes in the image in replace
def draw_bbox(image, bboxes,colors):
    """
    image is original image, read from cv2.imread
    bbox is x,y,x,y float numpy array
    
    """
    for bbox in bboxes:
        start_x,start_y,end_x,end_y = bbox[0:4].astype(int)
        confi_score = bbox[5]
        class_id = int(bbox[6])
        color = random.choice(colors)

        cv2.rectangle(image,(start_x,start_y),(end_x,end_y), color, thickness=2)
        label = "%s(%0.2f)"%(classes[class_id], confi_score) 
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = start_x + t_size[0] + 3, start_y + t_size[1] + 4
        cv2.rectangle(image, (start_x,start_y), c2,color, -1)
        cv2.putText(image, label, (start_x, start_y + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
 
    
    
def get_model(args):
     # set up the neural network
    print('Loading network....')
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print('Network successfully loaded')
    
    return model

def object_detect(img_path_list, bboxes_predict, ratio_list,colors):
    for i in range(len(img_path_list)):
        
        img = cv2.imread(img_path_list[i])
        
        img_w, img_h = img.shape[1], img.shape[0]
        
        ratio = ratio_list[i].numpy()
        predict_bbox = bboxes_predict[i]
        if predict_bbox.ndim == 1:
            predict_bbox = np.expand_dims(predict_bbox,axis=0)
        
        bbox = change_bboxes(predict_bbox, ratio, img_w,img_h)
        draw_bbox(img, bbox,colors)
        
        output_path = './det/det-%s'%(os.path.basename(img_path_list[i]))
        cv2.imwrite(output_path,img)

if __name__ == "__main__":
    args = arg_parse()
    images_path = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    classes = load_classes('data/coco.names')
    num_classes = len(classes)
    colors = pkl.load(open("./pallete", "rb"))

    model = get_model(args)
    
    model.net_info['height'] = args.reso
    inp_dim = int(model.net_info['height'])
    assert inp_dim %32 == 0
    assert inp_dim > 32
    
    if CUDA:
        model.cuda()

    # set model in evaluation mode
    model.eval()

    # Read the input images
    read_dir = time.time()

    try:
        img_list = [os.path.join(os.path.realpath('.'),images_path, img) for img in os.listdir(images_path) if img.endswith('jpg')]
    except NotADirectoryError:
        img_list = []
        img_list.append(os.path.join(os.path.realpath('.'), images_path))

    except FileNotFoundError:
        print('No file or directory with the name {}'.format(images_path))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    # create dataset and dataloader
    if args.exame:
        image_dataset = Image_dataset_test(img_list[0:3],letterbox_image,inp_dim=608)
    else:
        image_dataset = Image_dataset_test(img_list,letterbox_image,inp_dim=608)
    image_dataloader = DataLoader(image_dataset, batch_size= batch_size)
    print('Finished loading data')
    
    # get the predicted bboxes
    print('Predicting bboxes...')
    ratio_list = []
    bboxes_list = []
    img_path_list = []
    
    for batch in image_dataloader:
        resized_images, ratio, img_path = batch
        img_path_list += list(img_path)
        ratio_list += ratio
        predictions = model(resized_images,CUDA)
        bboxes = write_results(predictions, confidence=0.5,num_classes=80)

        bboxes_list += bboxes

    print('Drawing bboxes...')
    object_detect(img_path_list,bboxes_list,ratio_list,colors)
    print('\n Done \n')
    
    
    # List containing dimensions of original images
#    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
#    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
#
#    if CUDA:
#        im_dim_list = im_dim_list.cuda()
#
#    if batch_size != 1:
#        num_batches = len(imlist) // batch_size + leftover
#        im_batches = [torch.cat((im_batches[i*batch_size: min((i+1)*batch_size, len(im_batches)]))]
#

