# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import cv2
import numpy as np
import random
from PIL import Image
from numpy.random import rand, wald
from paddle.fluid.core import run_shell_command
from paddle.vision.transforms import ColorJitter
from ppocr.data.imaug.text_image_aug.augment_pro import tia_perspective, tia_stretch, tia_distort, WarpMLS


class RecAugPro(object):
    def __init__(self, use_tia=True, aug_prob=0.75, **kwargs):
        self.use_tia = use_tia
        self.aug_prob = aug_prob

    def __call__(self, data):
        # print('call  RecAugPro ---------------')
        img = data['image']
        img = warp(img, 10, self.use_tia, self.aug_prob)
        data['image'] = img
        return data


class ClsResizeImg(object):
    def __init__(self, image_shape, **kwargs):
        self.image_shape = image_shape

    def __call__(self, data):
        img = data['image']
        norm_img = resize_norm_img(img, self.image_shape)
        data['image'] = norm_img
        return data


class RecResizeImg(object):
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 character_type='ch',
                 **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_type = character_type

    def __call__(self, data):
        img = data['image']
        if self.infer_mode and self.character_type == "ch":
            norm_img = resize_norm_img_chinese(img, self.image_shape)
        else:
            norm_img = resize_norm_img(img, self.image_shape)
        data['image'] = norm_img
        return data


def resize_norm_img(img, image_shape):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32 * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def colorJitter(img):
    transform = ColorJitter(0.5, 0.5, 0.5, 0.5)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img)
    new_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    new_img = np.uint8(new_img)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    
    if h > 20 and w > 20:
        ks = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (ks, ks), 1)
    elif h > 10 and w > 10:
        ks = random.choice([3, 5])
        return cv2.GaussianBlur(img, (ks, ks), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=36):
    """
    Gasuss noise
    """
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


def marginJitter(image):
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



def erasing(image):
    img_h, img_w = image.shape[:2]
    line_width = int(min(img_h, img_w) * (random.random()*0.03+0.04))    # 0.04 - 0.07
    if line_width < 1:
        return image
    if random.random() < 0.5:
        line_sx = np.random.randint(0, img_w)
        line_sy = np.random.randint(0, img_h)
        line_ex = np.random.randint(0, img_w)
        line_ey = np.random.randint(0, img_h)

        image = cv2.line(image, (line_sx, line_sy), (line_ex, line_ey), (0, 0, 0), line_width)
    else:#加入竖条纹
        try:
            ws = random.sample(range(img_w), 5)
            for w in ws:
                image = cv2.line(image, (w, 0), (w, img_h-1), (0, 0, 0), line_width)
        except Exception as e:
            print(e)
            print('竖条纹 wrong')
                
    return image



class Config:
    """
    Config
    """
    def __init__(self, use_tia):
        self.use_tia = use_tia

    def make(self, w, h, ang):
        """
        make
        """
        self.perspective = self.use_tia
        self.stretch = self.use_tia
        self.distort = self.use_tia
        self.crop = True

        self.margin_jitter = True

        self.reverse = True
        self.noise = True
        self.jitter = True
        self.blur = True
        self.color = True
        self.pixel = True
        self.erasing = True
        self.flip = False
        self.gray = False


def warp(img, ang, use_tia=True, prob=0.75):
    """
    warp
    """
    h, w, _ = img.shape
    config = Config(use_tia=use_tia)
    config.make(w, h, ang)
    new_img = img

    # # pixel level perturbation
    if config.pixel and random.random() <= 1.0:
        new_img = pixelPert(new_img)

    if config.erasing and random.random() <= 0.5:
        new_img = erasing(new_img)

    # shape aug
    if random.random() < 0.7:
        if config.distort:
            img_height, img_width = img.shape[0:2]
            if random.random() <= prob and img_height >= 10 and img_width >= 10:
                new_img = tia_distort(new_img, random.randint(3, 6))
        if config.stretch:
            img_height, img_width = img.shape[0:2]
            if random.random() <= prob and img_height >= 10 and img_width >= 10:
                new_img = tia_stretch(new_img, random.randint(3, 6))
        if config.perspective:
            if random.random() <= prob:
                new_img = tia_perspective(new_img)
        if config.crop:
            img_height, img_width = img.shape[0:2]
            if random.random() <= prob and img_height >= 10 and img_width >= 10:
                new_img = get_crop(new_img)
    else:
        if random.random() <= prob:
            img_height, img_width = img.shape[0:2]
            if img_height >= 10 and img_width >= 10:
                new_img = marginJitter(new_img)


    # color aug
    if config.blur:
        if random.random() <= 0.5:
            new_img = blur(new_img)
    if config.color:
        new_img = colorJitter(new_img)
    if config.jitter:
        new_img = jitter(new_img)
    if config.noise:
        if random.random() <= 0.5:
            new_img = add_gasuss_noise(new_img)
    if config.reverse:
        if random.random() <= 0.5:
            new_img = 255 - new_img
    # if config.flip:
    #     if random.random() <= 0.5:
    #         try:
    #             if random.random() <= 0.5:
    #                 new_img = cv2.flip(new_img, 1)
    #             else:
    #                 new_img = cv2.flip(new_img, 0)
    #         except:
    #             print('flip wrong')
    # if config.gray:
    #     if random.random() <= 0.3:
    #         try:
    #             new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    #         except:
    #             print('gray wrong')
    return new_img
