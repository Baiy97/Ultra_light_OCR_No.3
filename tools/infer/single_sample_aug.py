import math
import cv2
import numpy as np
import random
from PIL import Image
from paddle.vision.transforms import ColorJitter
from ppocr.data.imaug.text_image_aug.augment_pro import tia_perspective, tia_stretch, tia_distort, WarpMLS

def colorJitter(img):
    transform = ColorJitter(0.2, 0.2, 0.2, 0.2)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img)
    new_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    new_img = np.uint8(new_img)
    return new_img


def marginJitter(image, margin_ratio=0.25):
    img_h, img_w = image.shape[:2]
    thresh = int(min(img_h, img_w) * 0.25)

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(-thresh, thresh+1), np.random.randint(-thresh, thresh+1)])
    dst_pts.append([img_w - np.random.randint(-thresh, thresh+1), np.random.randint(-thresh, thresh+1)])
    dst_pts.append([img_w - np.random.randint(-thresh, thresh+1), img_h - np.random.randint(-thresh, thresh+1)])
    dst_pts.append([np.random.randint(-thresh, thresh+1), img_h - np.random.randint(-thresh, thresh+1)])

    trans = WarpMLS(image, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def pixelPert(image):
    img_h, img_w = image.shape[:2]
    block_size = int(min(img_h, img_w) * 0.05)
    if block_size < 1:
        return image

    pert_num = int((img_h // block_size) * (img_w // block_size) * 0.05)

    if block_size < 1:
        return image

    for i in range(pert_num):
        start_x = random.randint(0, img_w-1)
        start_y = random.randint(0, img_h-1)
        end_x = start_x + block_size
        end_y = start_y + block_size

        direction = random.random()
        if direction < 0.25:
            # to top
            x1 = start_x
            y1 = start_y - block_size
            x2 = end_x
            y2 = start_y
            if min([start_x, end_x, x1, x2]) >= 0 and max([start_x, end_x, x1, x2]) < img_w and \
               min([start_y, end_y, y1, y2]) >= 0 and max([start_y, end_y, y1, y2]) < img_h:
                temp_patch = image[start_y:end_y, start_x:end_x]
                image[start_y:end_y, start_x:end_x] = image[y1:y2, x1:x2]
                image[y1:y2, x1:x2] = temp_patch
        elif direction < 0.5:
            # to bottom
            x1 = start_x
            y1 = end_y
            x2 = end_x
            y2 = end_y + block_size
            if min([start_x, end_x, x1, x2]) >= 0 and max([start_x, end_x, x1, x2]) < img_w and \
               min([start_y, end_y, y1, y2]) >= 0 and max([start_y, end_y, y1, y2]) < img_h:
                temp_patch = image[start_y:end_y, start_x:end_x]
                image[start_y:end_y, start_x:end_x] = image[y1:y2, x1:x2]
                image[y1:y2, x1:x2] = temp_patch
        elif direction < 0.75:
            # to left
            x1 = start_x - block_size
            y1 = start_y
            x2 = start_x
            y2 = end_y
            if min([start_x, end_x, x1, x2]) >= 0 and max([start_x, end_x, x1, x2]) < img_w and \
               min([start_y, end_y, y1, y2]) >= 0 and max([start_y, end_y, y1, y2]) < img_h:
                temp_patch = image[start_y:end_y, start_x:end_x]
                image[start_y:end_y, start_x:end_x] = image[y1:y2, x1:x2]
                image[y1:y2, x1:x2] = temp_patch
        else:
            # to right
            x1 = end_x
            y1 = start_y
            x2 = end_x + block_size
            y2 = end_y
            if min([start_x, end_x, x1, x2]) >= 0 and max([start_x, end_x, x1, x2]) < img_w and \
               min([start_y, end_y, y1, y2]) >= 0 and max([start_y, end_y, y1, y2]) < img_h:
                temp_patch = image[start_y:end_y, start_x:end_x]
                image[start_y:end_y, start_x:end_x] = image[y1:y2, x1:x2]
                image[y1:y2, x1:x2] = temp_patch

    return image


def sample_augmentation(img):
    '''
    TODO 8 types augmentation
    '''
    # raw image
    img_list = [img]


    '''
    TODO color reverse
    '''
    img_ = 255 - img.copy()
    img_list.append(img_)

    '''
    TODO color jitter
    '''
    img_ = colorJitter(img.copy())
    img_list.append(img_)

    '''
    TODO margin jitter
    '''
    img_ = marginJitter(img.copy(), margin_ratio=0.2)
    img_list.append(img_)

    img_ = marginJitter(img, margin_ratio=0.1)
    img_list.append(img_)

    '''
    TODO  perspective
    '''
    img_ = tia_perspective(img.copy(), ratio=0.2)
    img_list.append(img_)

    '''
    TODO  distort
    '''
    img_ = tia_distort(img.copy(), ratio=0.2)
    img_list.append(img_)

    '''
    TODO  stretch
    '''
    img_ = tia_stretch(img.copy(), ratio=0.2)
    img_list.append(img_)

    # '''
    # TODO pixel pert
    # '''
    # img_ = pixelPert(img.copy())
    # img_list.append(img_)


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
    



