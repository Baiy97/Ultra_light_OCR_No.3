
import collections
import json
import matplotlib.pyplot as plt
import cv2
import os
label_file = '/home/xingli/research/PaddleOCR/train/LabelTrain.txt'
img_dir = '/home/xingli/research/PaddleOCR/train/TrainImages'
with open(label_file, 'r') as f:
    lines = f.readlines()

ch2count = collections.defaultdict(int)
len2count = collections.defaultdict(int)
ratio2count = collections.defaultdict(int)
for line in lines:
    items = line.strip().split('\t')
    for ch in items[-1]:
        ch2count[ch] += 1
    
    len2count[len(items[-1])] += 1
    img = cv2.imread(os.path.join(img_dir, items[0]))
    h, w = img.shape[0], img.shape[1]
    ratio2count[round(w/h, 2)] += 1

l, c = zip(*list(ratio2count.items()))
# plt.xlabel('text length')
# plt.ylabel('number')
# plt.bar(l, c)
# plt.savefig('len2count.jpg')
# plt.close()
plt.xlabel('ratio')
plt.ylabel('number')
plt.bar(l, c)
plt.savefig('ratio2count.jpg')
plt.close()
save_dict = {'ch': ch2count, 'len': len2count, 'ratio': ratio2count}
with open('statics.json', 'w') as f:
    json.dump(save_dict, f, indent=4, ensure_ascii=False)

