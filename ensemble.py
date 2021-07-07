from collections import defaultdict

with open('details.txt', 'r') as f:
    lines = f.readlines()

'''
8 samples ensemble
'''
ccc = 0
final_res = []
for index in range(0, len(lines), 8):
    lines_ = lines[index:index+8]
    text_counter = defaultdict(int)
    length_counter = defaultdict(int)
    candidates = []
    text_score = []
    for line in lines_:
        # try:
        img_name, text, score1, score2, locs = line.strip().split('\t')
        if score1 == 'nan' or score2 == 'nan':
            score1 = 0.
            score2 = 0.
        else:
            score1, score2 = eval(score1), eval(score2) 
        
        if text.strip() == '':
            text = ' '
        else:
            text_1 = text.lstrip()
            text_2 = text_1.rstrip()
            if len(text_2) != len(text):
                score2 = score2[len(text)-len(text_1):]
                score2 = score2[:len(text_2)]
                score1 = sum(score2) / len(score2)
                text = text_2
        text_score.append([text, score1])
        text_counter[text] += 1
        length_counter[len(text)] += 1
        candidates.append([text, score1, score2])
        # except:
        #     print(line)
    if len(text_counter) == 0:
        final_res.append([img_name, ''])
        continue

    text_counter = sorted(text_counter.items(), key=lambda x: x[1])
    length_counter = sorted(length_counter.items(), key=lambda x: x[1])
    score_counter = sorted(text_score, key=lambda x: x[1])
    # import ipdb; ipdb.set_trace()
    # counter = sorted(text_score, key=lambda x: x[1])

    # cnt = 0
    # total_scores = 0
    # for item in candidates:
    #     if item[0] == text_counter[-1][0]:
    #         cnt += 1
    #         total_scores += item[1]
    # score_1 = total_scores / cnt

    # cancan = ''
    # cancan = text_counter[-1][0]
    final_res.append([img_name, text_counter[-1][0]])     # 81.99
    # final_res.append([img_name, score_counter[-1][0]])


    
    # if len(length_counter) == 1:
    #     # 每个位置投票
    #     textlen = length_counter[-1][0]
    #     chscore_counter = [defaultdict(float) for i in range(textlen)]
    #     chcnt_counter = [defaultdict(int) for i in range(textlen)]
    #     for item in candidates:
    #         if len(item[0]) == textlen:
    #             for ii, ch in enumerate(item[0]):
    #                 try:
    #                     chscore_counter[ii][ch] = item[2][ii] + chscore_counter[ii][ch]
    #                 except:
    #                     chscore_counter[ii][ch] = item[2] + chscore_counter[ii][ch]
    #                 chcnt_counter[ii][ch] += 1

    #     final_str = ''
    #     for ii in range(textlen):
    #         aa = sorted(chcnt_counter[ii].items(), key=lambda x: x[1])
    #         if aa[-1][1] > 4:
    #             final_str += aa[-1][0]
    #         else:
    #             chscore = dict()
    #             for key, value in chcnt_counter[ii].items():
    #                 chscore[key] = chscore_counter[ii][key] / value
    #             chscore = sorted(chscore.items(), key=lambda x: x[1])
    #             final_str += chscore[-1][0]
    #     # import ipdb; ipdb.set_trace()
    #     cancan = final_str
    # if text_counter[-1][1] >= 5:
    #     can_1 = text_counter[-1][0]
    #     can_2, score_2 = score_counter[-1]
    #     if can_2 == can_1:                              # 最高分结果 和 投票结果 相同
    #         cancan = can_1
    #     else:
    #         cnt = 0
    #         total_scores = 0
    #         for item in candidates:
    #             if item[0] == can_1:
    #                 cnt += 1
    #                 total_scores += item[1]
    #         score_1 = total_scores / cnt
    #         if score_1 + 0.1 > score_2:
    #             cancan = can_1
    #         else:
    #             cancan = can_2
    # else:
    #     if length_counter[-1][1] < 4:
    #         cancan = text_counter[-1][0]
    #     else:
    #         textlen = length_counter[-1][0]
    #         chscore_counter = [defaultdict(float) for i in range(textlen)]
    #         for item in candidates:
    #             if len(item[0]) == textlen:
    #                 for ii, ch in enumerate(item[0]):
    #                     try:
    #                         chscore_counter[ii][ch] = max(item[2][ii], chscore_counter[ii][ch])
    #                     except:
    #                         chscore_counter[ii][ch] = max(item[2], chscore_counter[ii][ch])
            
    #         final_str = ''
    #         for ii in range(textlen):
    #             chscore_counter[ii] = sorted(chscore_counter[ii].items(), key=lambda x: x[1])
    #             final_str += chscore_counter[ii][-1][0]
    #         # import ipdb; ipdb.set_trace()
    #         cancan = final_str
    
    # main_text = candidates[0][0]
    # if cancan[:-1] == main_text and cancan[-1] in ',;.，；。）)、:：':
    #     cancan = main_text
    # final_res.append([img_name, cancan])

'''
multi-level ensemble
'''
# pred_dict = dict()
# for line in lines:
#     img_name, pred, score = line.strip().split('\t')
#     if img_name not in pred_dict:
#         pred_dict[img_name] = []
#     if score == 'nan':
#         score = 0.
#     else:
#         score = eval(score)
#     pred_dict[img_name].append([pred, score])

# print(len(pred_dict))

# final_res = []
# for key, value in pred_dict.items():
#     if len(value) == 1:
#         final_res.append([key, value[0][0]])
#     elif len(value) == 2:
#         if value[0][1] > value[1][1]:
#             final_res.append([key, value[0][0]])
#         else:
#             final_res.append([key, value[1][0]])
#     else:
#         pred1, score1 = value[0][0], value[0][1]
#         pred2, score2 = value[1][0] + value[2][0], (value[1][1] + value[2][1]) / 2.

#         if score1 > score2:
#             final_res.append([key, pred1])
#         else:
#             final_res.append([key, pred2])


with open('ensemble_res.txt', 'w') as f:
    for item in final_res:
        f.write(item[0] + '\t' + item[1] + '\n')

