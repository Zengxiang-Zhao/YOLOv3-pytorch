from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5+ num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]


    # sigmoid the centre_x, centre_y, and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # add the centre offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset),1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors


    # sigmoid the classes
    prediction[:,:,5:5+num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))

    prediction[:,:,:4] *= stride


    return prediction


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:, 4]> confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask

    # covert the x,y,w,h to x,y,x,y
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)
    output = []

    for ind in range(batch_size):
        image_pred = prediction[ind] # one image

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score) # x,y,x,y, obj, class_conf, class number
        image_pred = torch.cat(seq,1)
        non_zero_ind = torch.nonzero(image_pred[:,4]).squeeze()
        try:
            image_pred_nonzero = image_pred[non_zero_ind]
        except:
            pass

        if image_pred_nonzero.shape[0] == 0:
            pass

        img_classes = torch.unique(image_pred_nonzero[:,-1]) # -1 index holds the class index, all the unique classes
        write = False
        bboxes_one_image = [] # to store the bbox 

        for cls in img_classes:
            image_pred_class = image_pred_nonzero[image_pred_nonzero[:,-1] == cls]

            #sort the class
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            num = len(image_pred_class)

            for i in range(num):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                #zero out all the detections that have IOU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask


                # change the image_pred_class
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind]
             
        
            if not write: # haven't initialize the bboxes_one_image
                bboxes_one_image = image_pred_class.numpy()
                write = True
            else:
                bboxes_one_image = np.vstack((bboxes_one_image,image_pred_class.numpy())) 
        
        output.append(bboxes_one_image)

    try:
        return output
    except:
        return 0

def bbox_iou(box1, box2):

    # get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:,1],box1[:, 2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:,1],box2[:, 2], box2[:,3]

    # get the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1,b2_x1)
    inter_rect_y1 = torch.max(b1_y1,b2_y1)
    inter_rect_x2 = torch.min(b1_x2,b2_x2)
    inter_rect_y2 = torch.min(b1_y2,b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1,min = 0)* torch.clamp(inter_rect_y2 - inter_rect_y1+1, min=0)

    # Union area
    b1_area = (b1_x2-b1_x1+1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2-b2_x1+1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area/(b1_area + b2_area - inter_area)

    return iou




def load_classes(namesfile):
    fp = open(namesfile,'r')
    names = fp.read().split('\n')[:-1]
    return names

def letterbox_image(img, inp_dim):
    """
    resize the image with unchanged aspect ratio using padding

    """

    img_h, img_w = img.shape[0], img.shape[1]
    w,h = inp_dim
    new_w = int(img_w* min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))

    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0],3),128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the nural network

    """
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    
    return img

