import binascii
import numpy as np
import os
# import cv2
import re
import csv
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional
import time

from PIL import Image

device = torch.device("cuda")
file_dir = []


def check_file(file_path):
    os.chdir(file_path)
    one_dir = os.path.abspath(os.curdir)
    file_dir.append(one_dir)
    all_file = os.listdir()
    files = []
    for f in all_file:
        if os.path.isdir(f):
            files.extend(check_file(file_path + '/' + f))
            os.chdir(file_path)
        else:
            files.append(f)
    return file_dir


def get_full_address():
    source_path = r"D:\桌面\codecsvv"
    file_list = check_file(source_path)
    for i, x in enumerate(file_list):
        all_files = os.listdir(x)
        add_list = []
        for filee in all_files:
            fullPath = x + '\\' + filee
            add_list.append(fullPath)
        return add_list


add = get_full_address()
add_name = [ite.split('\\')[3][:-4] for ite in add]
add_name.remove('ivy-2.0')


def split_string(content):
    content = re.split('([-,.(){}; "\n+])', content)
    return content


def comment_again(row_sentence):
    aa = split_string(row_sentence)
    if ('/*' in aa) and ('*/' in aa):
        aa = list(filter(None, aa))
        set_a = list(set(aa))
        length = len(set_a)
        count1 = set_a.count('/*')
        count2 = set_a.count('*/')
        count3 = set_a.count(' ')
        if count1 + count2 + count3 == length:
            return True
    return False


def pro_str(row_sentence):  # 处理/*  */     public static...
    a = split_string(row_sentence)
    if ('/*' in a) and ('*/' in a):
        a = list(filter(None, a))
        str_len = len(a)
        right = a.index('*/')
        inde = list(range(right + 1, str_len))
        new_a = []
        for i in inde:
            new_a.append(a[i])
        concat_str = ''.join(new_a)
        pattern = re.compile(r' {2,}')
        concat_str = re.sub(pattern, '', concat_str)
        concat_str = concat_str.strip()
        concat_str = '/* */ ' + concat_str
        return concat_str
    return row_sentence


def pro_str1(i):  # /* */  //
    texta = split_string(i)
    texta = list(filter(None, texta))
    if ('*/' in texta) and ('//' in texta) and not i.endswith('*/'):
        index1 = texta.index('*/')
        index = texta.index('//')
        list_text = texta[index1:index + 1]
        set_a = list(set(list_text))
        length = len(set_a)
        count2 = set_a.count('*/')
        count3 = set_a.count(' ')
        count4 = set_a.count('//')
        if count2 + count3 + count4 == length:
            return True
    return False


def pro_str2(i):  # /**字符串*/
    if i.startswith('/') and i.endswith('/'):
        return ''
    else:
        return i


def get_index_of_comment(index, text, text_len):
    count = 0
    for a in list(range(index, text_len)):
        if text[a] == '"':
            count += 1
    if count != 1:
        return index


def process_right_comment(text):
    # text = ' "file:///" + files[i].getAbsol//utePath()'
    if text.endswith('*/') and re.search(r'//', text) and not re.search(r'"', text):
        return text
    index = [i for i, j in enumerate(text) if j == '/' and j == text[i - 1] and i > 0]
    index = [get_index_of_comment(i, text, len(text)) for i in index]
    index = list(filter(None, index))
    if len(index) == 0:
        return text
    index = index[0]
    pro_text = text[:index - 1]
    return pro_text


def process_left_comment(text):
    if text.startswith('//'):
        return ''
    return text


def write(fulltext, name):
    path = 'C:\\桌面\\test\\' + name + '.txt'
    # if os.path.exists(path):
    #     os.remove(path)
    with open(path, 'a+', encoding='utf-8') as f:
        for i in fulltext:
            f.write(i)
    f.close()


def pro_str3(split_name):  # /** constructor inits everything and set up the search pattern
    nu1 = []  # 注释索引
    nu2 = []  # 注释索引
    for index, name in enumerate(split_name):
        # if name.startswith('/**') and len(name) > 3:
        if re.search(r'^/\*+', name):
            if index == 0:
                nu1.append(index)
                continue
            if split_name[index] != split_name[index - 2]:
                nu1.append(index)
                continue
        if name.endswith('*/') and re.search(r'\*+/$', name) and len(nu2) < len(nu1):
            # print(name)
            # print(len(nu1), len(nu2))
            nu2.append(index)
            if len(nu1) != len(nu2):
                print('索引长度不一')
                print(nu1)
                print(nu2)
                print(len(nu1))
                print(len(nu2))
                print(split_name[nu1[-2]:nu1[-2] + 3])
                print(split_name[nu2[-1]:nu2[-1] + 5])
                time.sleep(20)
    index = list(zip(nu1, nu2))
    comment_index = []
    for i in index:
        range_a = list(range(i[0], i[1] + 1))
        comment_index = comment_index + range_a
    text = [word for index, word in enumerate(split_name) if index not in comment_index]
    return text


def pro_str4(split_name):
    if_exist = re.search(r'/\*+', split_name)
    if if_exist and split_name.endswith('*/'):  # }/*  */
        return split_name[:if_exist.span()[0]]
    if_exist_right = re.search(r'\*+/', split_name)
    # split_name = split_name[if_exist.span()[1], if_exist_right.span()[0]]
    if if_exist and if_exist_right and (if_exist.span()[0] < if_exist_right.span()[0]):  # /*  */ }
        split_name = [j for i, j in enumerate(split_name) if
                      i not in range(if_exist.span()[0], if_exist_right.span()[1])]
        return ''.join(split_name).strip()
    return split_name


def pro_str5(source_code):
    source_code_a = []
    for index, i in enumerate(source_code):  # */和其他字符在一行
        if re.search(r'/\*+', i) and re.search(r'/\*+', source_code[index + 2]) and index <= len(
                source_code) - 2:
            if re.search(r'/\*+', i).span()[0] == re.search(r'/\*+', source_code[index + 2]).span()[0] == 0:
                continue
        if i.startswith('*/') and len(i) > 3:
            source_code_a.append('*/')
            source_code_a.append('\n')
            source_code_a.append(i[2:].strip())
            continue
        if list(filter(None, i.split(' '))).count('*/') == 1 and '"' not in i[  # fgsfsd */和其他字符在一行
                                                                            :re.search(r'\*/', i).span()[
                                                                                0]]:
            source_code_a.append(i[:re.search(r'\*/', i).span()[1]])
            source_code_a.append('\n')
            source_code_a.append(i[re.search(r'\*/', i).span()[1]:].strip())
            continue
        search1 = re.search(r'\*+/', i)
        search2 = re.search(r'//', i)  # }*/ //}}}
        if search1 and search2 and search1.span()[0] < search2.span()[0]:
            if list(set(list(filter(None, i[search1.span()[1]:search2.span()[0]].split(' '))))).count(
                    ' ') == 1 or len(
                list(filter(None, i[search1.span()[1]:search2.span()[0]].split(' ')))) == 0:
                source_code_a.append(i[:re.search(r'\*/', i).span()[1]])
                source_code_a.append('\n')
                source_code_a.append(i[re.search(r'\*/', i).span()[1]:].strip())
                continue
        source_code_a.append(i)
    return source_code_a





def step2_5(project_name, source_code):
    source_code = re.split('([\n])', source_code)
    source_code = [i if i == '\n' else i.strip() for i in source_code]  # 除去两端空格、\t
    if project_name.split('-')[0] == 'xerces':
        index_a = []
        index_b = []
        index_c = []
        for index, i in enumerate(source_code):
            if re.search(r'/\*{3,}', i) and not i.endswith('/'):
                index_a.append(index)
            if re.search(r'\*{3,}', i) and i.endswith('/') and (len(index_b) < len(index_a)):
                index_b.append(index)
                if len(index_a) != len(index_b):
                    print('xerces索引不一')
                    print(len(index_a))
                    print(len(index_b))
                    print(index_a)
                    print(index_b)
                    print(source_code[index_a[-2]:index_a[-2] + 3])
                    print(source_code[index_b[-1]:index_b[-1] + 5])
                    time.sleep(20)
            if len(index_a) == len(index_b) and re.search(r'/\*{3,}', i) and i.endswith(
                    '/') and index not in index_b:
                index_c.append(index)

        erase_index = []
        if len(index_a) != 0 and len(index_b) != 0:
            for item in list(zip(index_a, index_b)):
                erase_index = erase_index + list(range(item[0], item[1] + 1))
        if len(index_c) != 0:
            erase_index = erase_index + index_c
        source_code = [i for index, i in enumerate(source_code) if index not in erase_index]
    if project_name.split('-')[0] == 'jedit' or project_name.split('-')[0] == 'xerces':
        source_code = pro_str5(source_code)
    source_code = [pro_str2(i) for i in source_code]  # /**字符串*/
    source_code = [j for j in source_code if not pro_str1(j)]  # 除去/*  */   //
    source_code = [pro_str(j) for j in source_code]  # 处理/*  */     public static...
    source_code = [process_left_comment(j) for j in source_code]  # 处理 //单独一行
    source_code = [process_right_comment(j) for j in source_code]  # 处理 右边//
    source_code = [pro_str4(j) for j in source_code]  # 处理 右边带有/** The entry's permission mode. */
    # synapse、velocity、xalan去掉
    if project_name.split('-')[0] == 'ant' or project_name.split('-')[0] == 'log4j' or project_name.split('-')[
        0] == 'poi':
        source_code = pro_str5(source_code)

    source_code = pro_str3(source_code)  # /** constructor inits everything and set up the search pattern

    content = []
    # line_length = len(source_code_a)
    for idx, line in enumerate(source_code):
        if line == '\n':
            content.append('\n')
            continue
        content.append(line)

    content_a = []
    for line in content:
        line = split_string(line)
        content_a = content_a + line
    split_name = content_a
    # 去除''字符
    text = list(filter(None, split_name))
    exclude_enter = []  # 除去重复的回车
    for index, value in enumerate(text):
        if (text[index] == text[index - 1]) and index > 0 and value == '\n':
            continue
        exclude_enter.append(value)
    if exclude_enter[0] == '\n':
        exclude_enter = exclude_enter[1:]
    text = exclude_enter

    
    all_string_value = []
    for i in text:
        sum_string = 0
        for j in i:
            asc = ord(j)
            sum_string = sum_string + asc
        all_string_value.append(sum_string)
    
    return all_string_value, text



def max_min(x):
    x_max = np.max(x)
    x_min = np.min(x)
    x = [(i - x_min) / (x_max - x_min) for i in x]
    return x



def step5(file_name):
    # file_name = 'ant-1.3'
    project_name = file_name.split('-')[0]
    bug_file_addr_root = r'D:\桌面\bug-data'
    bug_file_addr_head = bug_file_addr_root + '\\' + project_name
    bug_file_addr = bug_file_addr_head + '\\' + file_name + '.csv'
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
            # label_name = label_name.split('.')[-2] + label_name.split('.')[-1]
            i += 1
            list_name.append(label_name)
        data_label = []
        data = []
        code_all = []
        single_data_len = []  # 统计每个instance token长度
        code_addr_head = r'D:\桌面\codecsvv'
        code_addr = code_addr_head + '\\' + file_name + '.csv'
        df = pd.read_csv(code_addr)
        for ita in list_name:  # 查找源代码文件是否存在
            try:
                index_code = df.loc[df['metric_name'] == ita].index[0]
                source_code = df.iloc[index_code]['file']  # 获取相应文件的源代码
                # except Exception as e:
                #     pass
                all_string_value_norm, single_code = step2_5(file_name, source_code)
                code_all.append(''.join(single_code))
                if len(all_string_value_norm) == 0:
                    single_name = df.iloc[index_code]['metric_name']
                    print(single_name)
                    print(source_code)
                    time.sleep(20)
                    df_a = pd.read_csv(bug_file_addr)
                    index_code = df_a.loc[df_a['name'] == single_name].index[0]
                    bug_count = df_a.iloc[index_code]['bug']
                    print(bug_count)
                    print('====================')
                #     name = df.iloc[index_code]['metric_name']
                #     print(name)
                #     all_string_value_norm = step2_5(file_name, source_code)
                #     print(all_string_value_norm)
                data.append(all_string_value_norm)
                single_data_len.append(len(all_string_value_norm))
                j += 1
                list_name_a.append(ita)
            except Exception as e:
                pass
        len_median = math.ceil(np.median(single_data_len))
        print('中位数：', len_median)
        # 用0以最长的序列长度填充其余序列
        max_len = max((len(l) for l in data))
        new_matrix = list(map(lambda l: list(l) + [0] * (max_len - len(l)), data))
        # 利用中位数截取序列
        median_new_matrix = []
        for row_new_matrix in new_matrix:
            median_new_matrix.append(torch.FloatTensor(row_new_matrix[:len_median]))
        print('标签个数:', i, '-', '实际代码个数:', j)
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
    return median_new_matrix, data_label, len_median, code_all


def validated(file_name, len_median):
    # file_name = 'ant-1.3'
    project_name = file_name.split('-')[0]
    bug_file_addr_root = r'D:\桌面\bug-data'
    bug_file_addr_head = bug_file_addr_root + '\\' + project_name
    bug_file_addr = bug_file_addr_head + '\\' + file_name + '.csv'
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
            # label_name = label_name.split('.')[-2] + label_name.split('.')[-1]
            i += 1
            list_name.append(label_name)
        data_label = []
        data = []
        single_data_len = []  # 统计每个instance token长度
        code_addr_head = r'D:\桌面\codecsvv'
        code_addr = code_addr_head + '\\' + file_name + '.csv'
        df = pd.read_csv(code_addr)
        for ita in list_name:  # 查找源代码文件是否存在
            try:
                index_code = df.loc[df['metric_name'] == ita].index[0]
                # except Exception as e:
                #     pass
                source_code = df.iloc[index_code]['file']  # 获取相应文件的源代码
                all_string_value_norm, _ = step2_5(file_name, source_code)
                data.append(all_string_value_norm)
                single_data_len.append(len(all_string_value_norm))
                # print(len(all_string_value_norm))
                # time.sleep(3)
                j += 1
                list_name_a.append(ita)
            except Exception as e:
                pass
        print('中位数：', len_median)
        # 用0以最长的序列长度填充其余序列
        max_len = max((len(l) for l in data))
        new_matrix = list(map(lambda l: list(l) + [0] * (max_len - len(l)), data))
        # 利用中位数截取序列
        median_new_matrix = []
        for row_new_matrix in new_matrix:
            median_new_matrix.append(torch.FloatTensor(row_new_matrix[:len_median]))
        print('标签个数:', i, '-', '实际代码个数:', j)
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
    return median_new_matrix, data_label, len_median


def get_cross_project_pair():
    cross_project = [['ant-1.6', 'camel-1.4'],
                     ['jedit-4.1', 'camel-1.4'],
                     ['camel-1.4', 'ant-1.6'],
                     ['poi-3.0', 'ant-1.6'],
                     ['camel-1.4', 'jedit-4.1'],
                     ['log4j-1.1', 'jedit-4.1'],
                     ['jedit-4.1', 'log4j-1.1'],
                     ['lucene-2.2', 'log4j-1.1'],
                     ['lucene-2.2', 'xalan-2.5'],
                     ['xerces-1.3', 'xalan-2.5'],
                     ['xalan-2.5', 'lucene-2.2'],
                     ['log4j-1.1', 'lucene-2.2'],
                     ['xalan-2.5', 'xerces-1.3'],
                     ['ivy-2.0', 'xerces-1.3'],
                     ['xerces-1.3', 'ivy-2.0'],
                     ['synapse-1.2', 'ivy-2.0'],
                     ['ivy-1.4', 'synapse-1.1'],
                     ['poi-2.5', 'synapse-1.1'],
                     ['ivy-2.0', 'synapse-1.2'],
                     ['poi-3.0', 'synapse-1.2'],
                     ['synapse-1.2', 'poi-3.0'],
                     ['ant-1.6', 'poi-3.0'],
                     ]
    return cross_project

