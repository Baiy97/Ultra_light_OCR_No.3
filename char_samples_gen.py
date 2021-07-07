import os
import cv2
import numpy as np
from tqdm import tqdm


image_dir = 'data/train_data/TrainImages/'
gt_txt = 'data/train_data/LabelTrain.txt'

gt_dict = dict()
with open(gt_txt, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        if len(line) > 1:
            gt_dict[line[0]] = line[1]
        else:
            gt_dict[line[0]] = ''
    
with open('details.txt', 'r') as f:
    lines = f.readlines()


char_gts = list()
for line in tqdm(lines):
    line = line.strip().split('\t')
    img_name, text, locs = line
    locs = eval(locs)

    # block size
    image = cv2.imread(image_dir + img_name)
    h, w, _ = image.shape
    textW = min(int(w / h * 32), 160)
    locs = locs[:int(textW / 160 * len(locs))]
    block_size = w / len(locs)

    # gt word
    gt_word = gt_dict[img_name]

    # 先不考虑空格
    scape_list = []
    prev = -1
    for i in range(len(locs)):
        if locs[i] == 0:
            continue
        if prev == -1:
            prev = int(i * block_size)
            continue
        
        if locs[i] != locs[i-1]:
            scape_list.append([prev, int(i*block_size)])
            prev = int(i*block_size)

    if prev != -1:
        scape_list.append([prev, w])

    if len(scape_list) == len(gt_word):
        '''
        TODO one by one
        '''
        for i in range(len(gt_word)):
            char_gts.append([img_name, gt_word[i], scape_list[i]])
    else:
        pass
        # if ' ' in gt_word and ' ' in text:
        #     '''
        #     split blank and compare one by one
        #     '''
        #     gt_word = gt_word.split()
        #     text = text.split()
        #     if len(gt_word) == len(text): 
        #         for i in range(len(gt_word)):
        #             if len(gt_word[i]) == len(text[i]):
        #                 for k in range(len(gt_word[i])):
        #                     char_gts.append([img_name, gt_word[i][k], scape_list[i][k]])


with open('data/train_data/LabelTrain_char_.txt', 'w') as f:
    for item in char_gts:
        f.write(item[0] + '\t' + item[1] + '\t' + str(item[2][0]) + ',' + str(item[2][1]) + '\n')

print('Finished...')


    
