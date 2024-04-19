import binascii
import numpy as np
import os
import cv2
import re
import csv
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional
import time
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics

from PIL import Image
from sklearn.manifold import TSNE
from torch.autograd import Variable

import network
import tesst


def encode(s):
    return ''.join([bin(ord(c)).replace('0b', '') for c in s])


def combine(source, target):
    # file_name = 'ant-1.3'
    def source_data(source):
        project_name = source.split('-')[0]
        bug_file_addr_root = r'D:\桌面\bug-data'
        bug_file_addr_head = bug_file_addr_root + '\\' + project_name
        bug_file_addr = bug_file_addr_head + '\\' + source + '.csv'
        with open(bug_file_addr, 'r') as csvFile:
            file1 = csv.reader(csvFile)
            row = [row for row in file1]
            head = row[0]
            row.remove(head)  # 去除第一行
            i = 0
            j = 0
            list_name = []  # 带标签的文件名
            list_name_a = []  # 源代码中的文件名
            for ita in row:
                label_name = ita[0]
                i += 1
                list_name.append(label_name)
            data_label = []
            data = []
            code_all = []
            single_data_len = []  # 统计每个instance token长度
            single_code_len = []  # 统计每个instance token长度
            code_addr_head = r'D:\桌面\codecsvv'
            code_addr = code_addr_head + '\\' + source + '.csv'
            df = pd.read_csv(code_addr)
            for ita in list_name:  # 查找源代码文件是否存在
                try:
                    index_code = df.loc[df['metric_name'] == ita].index[0]
                    source_code = df.iloc[index_code]['file']  # 获取相应文件的源代码
                    # except Exception as e:
                    #     pass
                    all_string_value_norm, single_code = tesst.step2_5(source, source_code)
                    concat_code = ''.join(single_code).encode('utf-8')
                    code_all.append(concat_code)
                    data.append(all_string_value_norm)
                    single_data_len.append(len(all_string_value_norm))
                    single_code_len.append(len(concat_code))
                    j += 1
                    list_name_a.append(ita)
                except Exception as e:
                    pass
            # median_src = math.ceil(np.median(single_data_len))
            median_src = math.ceil(np.median(single_code_len))
            print('中位数：', median_src)
            print('标签个数:', i, '-', '实际代码个数:', j)
            # 统计标签个数
            df_a = pd.read_csv(bug_file_addr)
            i = 0
            j = 0
            for single_name in list_name_a:
                index_code = df_a.loc[df_a['name'] == single_name].index[0]
                bug_count = df_a.iloc[index_code]['bug']
                if bug_count == 0:
                    data_label.append(torch.LongTensor([0]))
                    i += 1
                else:
                    data_label.append(torch.LongTensor([1]))
                    j += 1
            print('clean个数:', i, '-', 'bug个数:', j, 'bug ratio:{:.2}'.format(j / (i + j)))
        csvFile.close()
        return data, data_label, single_code_len, code_all

    def target_data(target):
        project_name = target.split('-')[0]
        bug_file_addr_root = r'D:\桌面\bug-data'
        bug_file_addr_head = bug_file_addr_root + '\\' + project_name
        bug_file_addr = bug_file_addr_head + '\\' + target + '.csv'
        with open(bug_file_addr, 'r') as csvFile:
            file1 = csv.reader(csvFile)
            row = [row for row in file1]
            head = row[0]
            row.remove(head)  # 去除第一行
            i = 0
            j = 0
            list_name = []  # 带标签的文件名
            list_name_a = []  # 源代码中的文件名
            for ita in row:
                label_name = ita[0]
                i += 1
                list_name.append(label_name)
            data_label = []
            data = []
            code_all = []
            single_data_len = []  # 统计每个instance token长度
            single_code_len = []  # 统计每个instance token长度
            code_addr_head = r'D:\桌面\codecsvv'
            code_addr = code_addr_head + '\\' + target + '.csv'
            df = pd.read_csv(code_addr)
            for ita in list_name:  # 查找源代码文件是否存在
                try:
                    index_code = df.loc[df['metric_name'] == ita].index[0]
                    source_code = df.iloc[index_code]['file']  # 获取相应文件的源代码
                    all_string_value_norm, single_code = tesst.step2_5(target, source_code)
                    concat_code = ''.join(single_code).encode('utf-8')
                    code_all.append(concat_code)
                    data.append(all_string_value_norm)
                    single_data_len.append(len(all_string_value_norm))
                    single_code_len.append(len(concat_code))
                    j += 1
                    list_name_a.append(ita)
                except Exception as e:
                    pass
            # median_tar = math.ceil(np.median(single_data_len))
            median_tar = math.ceil(np.median(single_code_len))
            print('中位数：', median_tar)
            print('标签个数:', i, '-', '实际代码个数:', j)
            # 统计标签个数
            df_a = pd.read_csv(bug_file_addr)
            i = 0
            j = 0
            for single_name in list_name_a:
                index_code = df_a.loc[df_a['name'] == single_name].index[0]
                bug_count = df_a.iloc[index_code]['bug']
                if bug_count == 0:
                    data_label.append(torch.LongTensor([0]))
                    i += 1
                else:
                    data_label.append(torch.LongTensor([1]))
                    j += 1
            print('clean个数:', i, '-', 'bug个数:', j, 'bug ratio:{:.2}'.format(j / (i + j)))
        csvFile.close()
        return data, data_label, single_code_len, code_all

    data_src, data_label_src, single_data_len_src, code_all_src = source_data(source)
    data_tar, data_label_tar, single_data_len_tar, code_all_tar = target_data(target)

    len_median = math.ceil(np.median(single_data_len_src))
    target_median = math.ceil(np.median(single_data_len_tar))
    combine_median = math.ceil(np.median(single_data_len_src + single_data_len_tar))
    if len_median < combine_median:
        len_median = combine_median
        print('使用融合方法，融合后中位数：', len_median)
    # # 用0以最长的序列长度填充其余序列
    max_len = max(single_data_len_src + single_data_len_tar)
    new_matrix_src = list(map(lambda l: l + b' ' * (max_len - len(l)), code_all_src))
    new_matrix_tar = list(map(lambda l: l + b' ' * (max_len - len(l)), code_all_tar))
    # # 利用中位数截取序列
    median_new_matrix_src = [i[:len_median] for i in new_matrix_src]
    median_new_matrix_tar = [i[:len_median] for i in new_matrix_tar]
    # median_new_matrix_src = [bytes(i, encoding='utf-8') for i in median_new_matrix_src]
    # median_new_matrix_tar = [bytes(i, encoding='utf-8') for i in median_new_matrix_tar]
    return median_new_matrix_src, data_label_src, median_new_matrix_tar, data_label_tar, len_median


def get_images(source, target):#对于图像进行图形增强
    print(source, target)
    # source = 'ant-1.3'
    # target = 'ant-1.4'
    # new_matrix3, data_label3, len_median3, code_all = tesst.step5(source)
    new_matrix3, data_label3, new_matrix4, data_label4, len_median3 = combine(source, target)

    def gener_img(item):
        content = item
        # print(content)
        hexst = binascii.hexlify(content)  # 将二进制数以十六进制形式展现
        # print(hexst)
        fh = np.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])  # 将十六进制以十进制形式展现
        # print(fh[:12])
        #end = len(fh) - len(fh) % 3
        end = len(fh) - len(fh) % 4
        # g = fh[0:end:3]
        # r = fh[1:end:3]
        # b = fh[2:end:3]
        g = fh[0:end:4]
        r = fh[1:end:4]
        b = fh[2:end:4]
        alpha = fh[3:end:4]
        maxalpha = max(alpha)
        minaplha = min(alpha)
        guiyialpha = (alpha[:]-minaplha)/(maxalpha-minaplha)
        print(guiyialpha[:5])
        print(len(g), len(r), len(b),len(guiyialpha))
        # img2 = cv2.merge([guiyialpha*b, guiyialpha*r, guiyialpha*g])
        img2 = cv2.merge([(1-guiyialpha)+r*guiyialpha,(1-guiyialpha)+g*guiyialpha,(1-guiyialpha)+b*guiyialpha])
        # print(img2)
        # print(img2.shape)
        width = int(np.sqrt(len(b)))
        # img1 = img2[:len(b) - len(b) % width]
        img1 = img2[:width * width]
        # img = np.reshape(img1, (width, len(b) // width, 3))
        # img = np.reshape(img1, (width, width, 3))#保存图片专用的一个方法
        img = np.reshape(img1, (3, width, width)) #用于跑实验用的东西

        # path_save = r'D:\data\pic\b.png'
        # cv2.imwrite(path_save, img)
        # path_save = r'D:\a.png'
        # cv2.imwrite(path_save, img)

        # img.save(path_save)
        # cv2.imshow('a', img)
        # cv2.waitKey(0)

        img = torch.FloatTensor(img)
        # time.sleep(20)
        return img

    img_all_src = [gener_img(i) for i in new_matrix3]
    img_all_tar = [gener_img(i) for i in new_matrix4]

    x = torch.stack(img_all_src)
    y = torch.stack(img_all_tar)
    data_label3 = torch.stack(data_label3)
    data_label4 = torch.stack(data_label4)
    return x, data_label3, y, data_label4

def tsne(df_fea, df_lab):
    # print(df_fea, df_lab)
    tsne = TSNE(2, 38, 20, 600)

    result = tsne.fit_transform(df_fea)

    colors = ['c', 'r']
    idx_1 = [i1 for i1 in range(len(df_lab)) if df_lab[i1] == 0]
    fig1 = plt.scatter(result[idx_1, 0], result[idx_1, 1], 20, color=colors[0], label='Clean')

    idx_2 = [i2 for i2 in range(len(df_lab)) if df_lab[i2] == 1]
    fig2 = plt.scatter(result[idx_2, 0], result[idx_2, 1], 20, color=colors[1], label='Defective')

from transformer import ViT
def get_image():
    base_network = ViT()
    # print(base_network.features.)
    # time.sleep(20)
    # classifier_layer = nn.Linear(base_network.output_num(), 1)
    # classifier_layer = Classifier(base_network.output_num())
    # print(classifier_layer)

    # classifier_layer.weight.data.normal_(0, 0.01)
    # classifier_layer.bias.data.fill_(0.0)
    if torch.cuda.is_available():
        # classifier_layer = classifier_layer.cuda()
        base_network = base_network.cuda()
    parameter_list = [{"params": base_network.parameters(), "lr": 10},
                      ]
    import torch.optim as optim
    optimizer = optim.SGD(parameter_list,
                          **({"momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}))
    # print(optimizer)
    # time.sleep(20)
    # from itertools import chain
    # optimizer = optim.Adam(chain(base_network.parameters(), classifier_layer.parameters()), lr=10)
    # print(optimizer)
    # time.sleep(20)
    import lr_schedule
    lr_scheduler = lr_schedule.schedule_dict['inv']  # inf
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    source = 'ant-1.6'
    target = 'camel-1.4'
    # source = 'ant-1.6'
    # target = 'camel-1.4'
    x, data_label3, y, data_label4 = get_images(source, target)
    # x = torch.randn(124, 3, 25, 25)
    loss_all = []
    fscore_all = []
    for i in range(400):
        optimizer = lr_scheduler(param_lr, optimizer, i, **{"init_lr": 0.0003, "gamma": 0.0003, "power": 0.75})
        # print(optimizer)
        # time.sleep(20)
        optimizer.zero_grad()

        base_network.train()
        # classifier_layer.train()
        features = base_network(x.cuda())
        # outputs = classifier_layer(features)
        outputs = features
        loss = F.binary_cross_entropy_with_logits(outputs, data_label3.cuda().float())
        loss_all.append(loss)
        loss.backward()
        optimizer.step()

        base_network.eval()
        # classifier_layer.eval()
        features = base_network(y.cuda())
        # y_pred = classifier_layer(features).detach().cpu().numpy().flatten()
        y_pred = features.detach().cpu().numpy().flatten()
        y_test = data_label4
        y_test = [ele.numpy()[0].astype('float32') for ele in y_test]
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
        fscore = (2 * precision * recall) / (precision + recall + 10 ** -10)
        ix = np.argmax(fscore)
        y_pred_class = [x > thresholds[ix] for x in y_pred]

        precision, recall, AUC = metrics.precision_score(y_test, y_pred_class), \
                                 metrics.recall_score(y_test, y_pred_class), \
                                 metrics.roc_auc_score(y_test, y_pred)
        fscore_single = fscore[ix]
        fscore_all.append(fscore_single)
    import matplotlib.pyplot as plt
    sum = 0
    for i in fscore_all:
        sum = sum + i
    fscore_ave = sum / 400
    print('fscore_ave', fscore_ave)

    print('fscore-max:', np.max(fscore_all))
    #输出对应的recall、accuracy、precision
    print('recall-max:', np.max(recall))
    print('AUC-max:', np.max(AUC))
    print('precision-max:', np.max(precision))
    plt.figure()
    x = np.arange(len(loss_all))
    plt.plot(x, loss_all, label='loss')
    plt.plot(x, fscore_all, label='fscore')
    plt.legend(loc='upper right')
    plt.show()
    # print(lr_scheduler)
    # print(classifier_layer)

    outputs = features.data.float()

    labels = data_label3.data.float()
    outputs = outputs.cpu().numpy()
    labels = labels.cpu().numpy()
    x_tr1 = np.array(y).reshape(-1, 1)
    # x_ts1 = np.array(y_pred_class).reshape(-1, 1)
    tsne(outputs, labels)
    # tsne(x,data_label3)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()

get_image()





