import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re



# 原始的分类变量
pos_tags = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg', 'd', 'dc', 'df', 'e', 'f', 'g', 'h', 'i', 'ia', 'ib', 'id', 'in', 'iv', 'j', 'ja', 'jb', 'jd', 'jn', 'jv', 'k', 'l', 'la', 'lb', 'ld', 'ln', 'lv', 'm', 'mq', 'Ng', 'n', 'nr', 'nrf', 'nrg', 'ns', 'nt', 'nx', 'nz', 'o', 'p', 'Qg', 'q', 'qb', 'qc', 'qd', 'qe', 'qj', 'ql', 'qr', 'qt', 'qv', 'qz', 'Rg', 'r', 'rr', 'ry', 'ryw', 'rz', 'rzw', 's', 'Tg', 't', 'tt', 'u', 'ui', 'ul', 'uo', 'us', 'uz', 'ud', 'ue', 'vt', 'Vg', 'v', 'vd', 'vi', 'vl', 'vn', 'vq', 'vu', 'vx', 'w', 'wd', 'wf', 'wj', 'wk', 'wky ', 'wkz', 'wm', 'wp', 'ws', 'wt', 'wu', 'ww', 'wy', 'wyy', 'wyz', 'x', 'y', 'z']

# 创建LabelEncoder对象，将分类变量映射到整数
label_encoder = LabelEncoder()
encoded_pos_tags = label_encoder.fit_transform(pos_tags)

# 创建OneHotEncoder对象，进行独热编码
onehot_encoder = OneHotEncoder(categories='auto')
encoded_pos_tags = encoded_pos_tags.reshape(-1, 1)
onehot_encoded_pos_tags = onehot_encoder.fit_transform(encoded_pos_tags).toarray()

# 生成映射字典
original_classes = label_encoder.classes_
onehot_categories = onehot_encoder.categories_[0]
mapping_dict = dict(zip(original_classes, onehot_encoded_pos_tags))

line = '迈向/vt  充满/vt  希望/n  的/ud  新/a  世纪/n  ——/wp  一九九八年/t  新年/t  讲话/n  （/wkz  附/vt  图片/n  １/m  张/qe  ）/wky  '

# 找到第一个汉字的索引
first_chinese_index = next((i for i, char in enumerate(line) if '\u4e00' <= char <= '\u9fff'), None)

# 找到第一个汉字后的斜杠索引
slash_indices = [i for i, char in enumerate(line[first_chinese_index:]) if char == "/"]
result = []
results = np.array([])
pronoun_index_front = 7
index_front = 3
index_behind = 4
# 遍历斜杠索引
for start_index in slash_indices:
            substring = ""
            new_substring = ""
            index = first_chinese_index + start_index + 1
            i = 0  # 在进入一句换的遍历的开始阶段将i初始化为0，这时候i指向的是第0个字
            if '\u4e00' < line[index - 2] < '\u9fff':
                if i < pronoun_index_front:
                    # 跳过第一个斜杠后的字符
                    if index < len(line):
                        # 遍历直到遇到空格或结尾
                        while index < len(line) and line[index] != " ":
                            substring += line[index]
                            index += 1

                        # 检查子字符串是否在映射字典中
                        if substring != "":
                            onehot_encoding = mapping_dict.get(substring)

                            # 在这里添加下一步的代码，计算其他特征值

                            # 继续向后遍历直到再次遇到汉字后的斜杠
                            while index < len(line):
                                if line[index] == "/" and '\u4e00' <= line[index - 1] <= '\u9fff':
                                    break
                                index += 1
                            index += 1
                            if index < len(line):
                                # 遍历直到遇到空格或结尾
                                while index < len(line) and line[index] != " ":
                                    new_substring += line[index]
                                    index += 1
                                    # 检查子字符串是否在映射字典中
                                if new_substring != "":
                                    new_onehot_encoding = mapping_dict.get(new_substring)
                                    if onehot_encoding is not None and new_onehot_encoding is not None:
                                        # 获取当前循环的字典中的 pronoun_index_front

                                        # 计算 length
                                        length = pronoun_index_front - i
                                        length_array = np.expand_dims(length, axis=0)

                                        if index_front <= i <= index_behind:
                                            tof = 1
                                        else:
                                            tof = 0
                                        tof_array = np.expand_dims(tof, axis=0)
                                        result_list = np.concatenate(
                                            (onehot_encoding, new_onehot_encoding, length_array, tof_array))
                                        i = i + 1
                                        if results.size == 0:
                                            # 如果结果矩阵为空，直接将结果赋值给 results
                                            results = result_list
                                        else:
                                            # 如果结果矩阵不为空，使用 np.vstack() 将结果添加到矩阵中
                                            results = np.vstack((results, result_list))



# 在循环结束后，一起输出结果
for result in results:
    print(result)
print("Over!")