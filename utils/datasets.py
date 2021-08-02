import glob
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

from utils.utils import xyxy2xywh,letterbox_image


# for detect
class Image_dataset_detect(Dataset):
    def __init__(self,imgs_path,inp_dim, debug=False):
        super(Image_dataset_detect,self).__init__()
        img_formats = ['.jpg', '.jpeg', '.png', '.tif']
        assert os.path.isdir(imgs_path), f'the path {imgs_path} is not dir'
        self.imgs_path = imgs_path
        # create imgs_list and labels_list
        self.imgs_list = os.listdir(self.imgs_path)
        self.imgs_list = [x for x in self.imgs_list if os.path.splitext(x)[-1].lower() in img_formats]

        if debug:
            self.imgs_list = self.imgs_list[:3]

        self.inp_dim = inp_dim
        
    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self,indx):
        # get image
        img_path = os.path.join(self.imgs_path, self.imgs_list[indx] )
        img = cv2.imread(img_path)

        assert img  is not None, f'Could not read the image in [{img_path}]' 

        resized_image, ratio = letterbox_image(img,self.inp_dim)
#         resized_image = resized_image[:,:,::-1]
        return resized_image,ratio,img_path

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, imgs_path,labels_path, img_size=416, augment=False, debug=False):
        img_formats = ['.jpg', '.jpeg', '.png', '.tif']
        self.imgs_path = imgs_path
        self.labels_path = labels_path

        assert os.path.isdir(imgs_path), f'{imgs_path} : is not a dictionary'
        assert os.path.isdir(labels_path), f'{labels_path}: is not a dictionary'

        # create imgs_list and labels_list
        self.imgs_list = os.listdir(imgs_path)
        self.imgs_list = [x for x in self.imgs_list if os.path.splitext(x)[-1] in img_formats]

        self.labels_list = self.label_files = [
            x.replace('.bmp', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
            for x in self.imgs_list]

        assert len(self.imgs_list) > 0, 'No images found in %s' % imgs_path
        if debug:
            self.imgs_list = self.imgs_list[:4]
            self.labels_list = self.labels_list[:4]

        self.img_size = img_size
        self.augment = augment


    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img_path =  os.path.join(self.imgs_path, self.imgs_list[index])
        label_path = os.path.join(self.labels_path, self.labels_list[index])

        img = cv2.imread(img_path)  # BGR
        assert img is not None, 'File Not Found ' + img_path

        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50  # must be < 1.0
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value

            a = (random.random() * 2 - 1) * fraction + 1
            b = (random.random() * 2 - 1) * fraction + 1
            S *= a
            V *= b

            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=self.img_size, mode='square')

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            with open(label_path, 'r') as file:
                lines = file.read().splitlines()

            x = np.array([x.split() for x in lines], dtype=np.float32)
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio * w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = ratio * h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = ratio * w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = ratio * h * (x[:, 2] + x[:, 4] / 2) + padh

        # Augment image and labels
        if self.augment:
            img, labels = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) / self.img_size

        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() > 0.5:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() > 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Normalize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return torch.from_numpy(img), labels_out, img_path, (h, w)

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw
        # img = [C,H,W],using torch.stack(img,0),using torch.stack(img,0) we could get [batch, C,H,W]


def letterbox(img, height=416, color=(127.5, 127.5, 127.5), mode='rect'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]

    # Select padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'rect':  # rectangle
        dw = np.mod(height - new_shape[0], 32) / 2  # width padding
        dh = np.mod(height - new_shape[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (height - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


def random_affine(img, targets=(), degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 1:5].copy()
        area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction of bounding boxes
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        np.clip(xy, 0, height, out=xy)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return imw, targets


def convert_images2bmp():
    # cv2.imread() jpg at 230 img/s, *.bmp at 400 img/s
    for path in ['../coco/images/val2014/', '../coco/images/train2014/']:
        folder = os.sep + Path(path).name
        output = path.replace(folder, folder + 'bmp')
        if os.path.exists(output):
            shutil.rmtree(output)  # delete output folder
        os.makedirs(output)  # make new output folder

        for f in tqdm(glob.glob('%s*.jpg' % path)):
            save_name = f.replace('.jpg', '.bmp').replace(folder, folder + 'bmp')
            cv2.imwrite(save_name, cv2.imread(f))

    for label_path in ['../coco/trainvalno5k.txt', '../coco/5k.txt']:
        with open(label_path, 'r') as file:
            lines = file.read()
        lines = lines.replace('2014/', '2014bmp/').replace('.jpg', '.bmp').replace(
            '/Users/glennjocher/PycharmProjects/', '../')
        with open(label_path.replace('5k', '5k_bmp'), 'w') as file:
            file.write(lines)
