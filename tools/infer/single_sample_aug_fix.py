import math
import cv2
import numpy as np
import random
from PIL import Image
from paddle.vision.transforms import ColorJitter
from ppocr.data.imaug.text_image_aug.warp_mls import WarpMLS


def tia_perspective(src, ratio=0.25, flag=1):
    img_h, img_w = src.shape[:2]

    # thresh = img_h // 2
    thresh = min(img_w, img_h) * ratio

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])


    dd = - int(thresh * 0.5)

    if flag == 1:
        dst_pts.append([0, dd])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h - dd])
    elif flag == 2:
        dst_pts.append([0, 0])
        dst_pts.append([img_w, dd])
        dst_pts.append([img_w, img_h - dd])
        dst_pts.append([0, img_h])
    else:
        dst_pts.append([0, dd])
        dst_pts.append([img_w, dd])
        dst_pts.append([img_w, img_h - dd])
        dst_pts.append([0, img_h - dd])


    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def marginJitter(image, margin_ratio=0.25, positive=1):
    img_h, img_w = image.shape[:2]
    thresh = int(min(img_h, img_w) * 0.25)

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dd = int(positive * thresh * 0.5)
    dst_pts.append([dd, dd])
    dst_pts.append([img_w - dd, dd])
    dst_pts.append([img_w - dd, img_h - dd])
    dst_pts.append([dd, img_h - dd])

    trans = WarpMLS(image, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def sample_augmentation(img):
    '''
    TODO 8 types augmentation
    '''
    # raw image
    img_list = [img]


    '''
    TODO color reverse * 1
    '''
    img_ = 255 - img.copy()
    img_list.append(img_)

    '''
    TODO margin jitter * 2
    '''
    img_ = marginJitter(img.copy(), margin_ratio=0.1, positive=1)
    img_list.append(img_)

    img_ = marginJitter(img.copy(), margin_ratio=0.05, positive=1)
    img_list.append(img_)

    img_ = 255 - img.copy()
    img_ = marginJitter(img_, margin_ratio=0.1, positive=1)
    img_list.append(img_)

    '''
    TODO  perspective * 3
    '''
    img_ = tia_perspective(img.copy(), ratio=0.1, flag=1)
    img_list.append(img_)

    img_ = tia_perspective(img.copy(), ratio=0.1, flag=2)
    img_list.append(img_)

    img_ = tia_perspective(img.copy(), ratio=0.1, flag=3)
    img_list.append(img_)

    return img_list




# split_dict = dict()
# with open('test_seq_char_split.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip().split('\t')
#         try:
#             img_name, pred = line[0], line[1]
#             split_dict[img_name] = [pred, line[2:]] 
#         except:
#             img_name = line[0]
#             split_dict[img_name] = ['', []]  



'''
TODO 读取 seq char split 然后做一些切割
'''
def split_aug(img, img_name):
    h, w, _ = img.shape

    split_items = split_dict[img_name.split('/')[-1]]
    pred, _, _, locs = split_items
    if pred == '':
        return [img_name], [img]
    
    img_list = [img]
    if len(pred) == 1:
        sx = max(0, eval(locs[0][2:].split(',')[0]))
        ex = min(img.shape[1], eval(locs[0][2:].split(',')[1]))
        thresh = int(min(h * 0.07, (ex-sx) * 0.07))
        sx = max(0, sx - thresh)
        ex = min(w, ex - thresh)
        img_ = img[:, sx:ex].copy()
        img_list.append(img_)
        return [img_name] * 2, img_list
    else:
        # left part
        sx, ex = 0, len(locs)//2-1
        sx = max(0, eval(locs[sx][2:].split(',')[0]))
        ex = min(img.shape[1], eval(locs[ex][2:].split(',')[1]))
        thresh = int(min(h * 0.07, (ex-sx) * 0.07))
        sx = max(0, sx - thresh)
        ex = min(w, ex - thresh)
        img_ = img[:, sx:ex].copy()
        img_list.append(img_)

        # right part
        sx, ex = len(locs)//2, -1
        sx = max(0, eval(locs[sx][2:].split(',')[0]))
        ex = min(img.shape[1], eval(locs[ex][2:].split(',')[1]))
        thresh = int(min(h * 0.07, (ex-sx) * 0.07))
        sx = max(0, sx - thresh)
        ex = min(w, ex - thresh)
        img_ = img[:, sx:ex].copy()
        img_list.append(img_)

        return [img_name] * 3, img_list
    



