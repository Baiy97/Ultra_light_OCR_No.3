import os


res_dict = dict()
with open('res_8193.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        if len(line) > 1:
            res_dict[line[0]] = line[1]
        else:
            res_dict[line[0]] = ''

res_score1_dict = dict()
with open('ensemble_res.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        if len(line) > 1:
            res_score1_dict[line[0]] = line[1]
        else:
            res_score1_dict[line[0]] = ''


diff_dict = dict()
cnt = 0
for key in res_dict:
    if res_dict[key] == res_score1_dict[key]:
        cnt +=1
        continue
    diff_dict[key] = [res_dict[key], res_score1_dict[key]]
    print(key)
    print(res_dict[key])
    print(res_score1_dict[key])
    print('======================')

print(cnt / 10000) 
import ipdb; ipdb.set_trace()

    
