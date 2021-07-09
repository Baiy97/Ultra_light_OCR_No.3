"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import six
import cv2
import numpy as np
import random


class DecodeImage(object):
    """ decode image """

    def __init__(self, img_mode='RGB', channel_first=False, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img

        label = data['label']
        if isinstance(label, list):
            if len(label) == 2 and len(label[1].split(',')) == 2:
                raise Exception("Not Single char training..")
                # '''
                # single char train
                # '''
                # locs = eval(label[1])
                # scape = locs[1] - locs[0]
                # # do jitter
                # thresh = scape * 0.1
                # if thresh > 1:
                #     sx = max(0, locs[0] + np.random.randint(-thresh, 0.3*thresh+1))
                #     ex = min(img.shape[1], locs[1] + np.random.randint(-thresh, 0.3*thresh+1))
                #     img = img[:, sx:ex].copy()
                #     data['image'] = img
                # data['label'] = label[0]
            else:
                '''
                multi-level train
                需要跳跃连接
                '''
                full_trans_ = label[0]
                sub_trans_ = label[1:]
                if random.random() < 0.65:   # 连续
                    sx = np.random.randint(0, len(sub_trans_)+1)
                    ex = np.random.randint(0, len(sub_trans_)+1)
                    sx, ex = min(sx, ex), max(sx, ex)
                    if sx == ex or ex - sx == len(full_trans_):
                        data['label'] = full_trans_
                    else:
                        sub_trans = sub_trans_[sx:ex]
                        sub_label = ''.join([t[0] for t in sub_trans])
                        sub_label = sub_label.strip()      # 去掉两端的空格
                        sx = max(0, eval(sub_trans[0][2:].split(',')[0]))       # why [2:], 有可能是',,0,13'这种情况
                        ex = min(img.shape[1], eval(sub_trans[-1][2:].split(',')[1]))
                        # do jitter
                        h, w, _ = img.shape
                        thresh = h * 0.1
                        sx = max(0, sx + np.random.randint(-thresh, 0.3*thresh+1))
                        ex = min(w, ex + np.random.randint(-thresh, 0.3*thresh+1))
                        img = img[:, sx:ex].copy()
                        data['image'] = img
                        data['label'] = sub_label
                else:
                    xs = [np.random.randint(0, len(sub_trans_)+1) for i in range(4)]
                    xs.sort()
                    # part1
                    sx = xs[np.random.randint(0, 4)]    # 0,1,2,3,4
                    ex = xs[np.random.randint(0, 4)]
                    sx, ex = min(sx, ex), max(sx, ex)
                    if sx == ex or ex - sx == len(full_trans_):
                        sub_label_1 = full_trans_
                        img_1 = img.copy()
                    else:
                        sub_trans = sub_trans_[sx:ex]
                        sub_label = ''.join([t[0] for t in sub_trans])
                        sub_label_1 = sub_label.lstrip()      # 去掉左侧的空格
                        sx = max(0, eval(sub_trans[0][2:].split(',')[0]))       # why [2:], 有可能是',,0,13'这种情况
                        ex = min(img.shape[1], eval(sub_trans[-1][2:].split(',')[1]))
                        # do jitter
                        h, w, _ = img.shape
                        thresh = h * 0.1
                        sx = max(0, sx + np.random.randint(-thresh, 0.3*thresh+1))
                        ex = min(w, ex + np.random.randint(-thresh, 0.3*thresh+1))
                        img_1 = img[:, sx:ex].copy()

                    # part2
                    sx = xs[np.random.randint(0, 4)]
                    ex = xs[np.random.randint(0, 4)]
                    sx, ex = min(sx, ex), max(sx, ex)
                    if sx == ex or ex - sx == len(full_trans_):
                        sub_label_2 = full_trans_
                        img_2 = img.copy()
                    else:
                        sub_trans = sub_trans_[sx:ex]
                        sub_label = ''.join([t[0] for t in sub_trans])
                        sub_label_2 = sub_label.rstrip()      # 去掉右侧的空格
                        sx = max(0, eval(sub_trans[0][2:].split(',')[0]))       # why [2:], 有可能是',,0,13'这种情况
                        ex = min(img.shape[1], eval(sub_trans[-1][2:].split(',')[1]))
                        # do jitter
                        h, w, _ = img.shape
                        thresh = h * 0.1
                        sx = max(0, sx + np.random.randint(-thresh, 0.3*thresh+1))
                        ex = min(w, ex + np.random.randint(-thresh, 0.3*thresh+1))
                        img_2 = img[:, sx:ex].copy()
                        # concat part1 & part2
                    sub_label = sub_label_1 + sub_label_2
                    img = np.concatenate((img_1, img_2), axis=1)
                    data['image'] = img
                    data['label'] = sub_label


        if 'comp' in data:
            # TODO concat
            img_path, label = data['comp']
            with open(img_path, 'rb') as f:
                img = f.read()

            comp_data = {'img_path':img_path, 'label':label, 'image':img}
            comp_data = self.__call__(comp_data)
            comp_image = comp_data['image']
            comp_label = comp_data['label']

            h1, w1, _ = data['image'].shape
            h2, w2, _ = comp_image.shape
            if h1 > h2:
                comp_image = cv2.resize(comp_image, (int(h1/h2*w2), h1))
            else:
                data['image'] = cv2.resize(data['image'], (int(h2/h1*w1), h2))

            data['label'] = data['label'] + comp_label
            data['image'] = np.concatenate((data['image'], comp_image), axis=1)

        return data


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class E2EResizeForTest(object):
    def __init__(self, **kwargs):
        super(E2EResizeForTest, self).__init__()
        self.max_side_len = kwargs['max_side_len']
        self.valid_set = kwargs['valid_set']

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        if self.valid_set == 'totaltext':
            im_resized, [ratio_h, ratio_w] = self.resize_image_for_totaltext(
                img, max_side_len=self.max_side_len)
        else:
            im_resized, (ratio_h, ratio_w) = self.resize_image(
                img, max_side_len=self.max_side_len)
        data['image'] = im_resized
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_for_totaltext(self, im, max_side_len=512):

        h, w, _ = im.shape
        resize_w = w
        resize_h = h
        ratio = 1.25
        if h * ratio > max_side_len:
            ratio = float(max_side_len) / resize_h
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image(self, im, max_side_len=512):
        """
        resize image to a size multiple of max_stride which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(max_side_len) / resize_h
        else:
            ratio = float(max_side_len) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)
