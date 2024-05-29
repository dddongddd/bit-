import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
from tqdm import tqdm

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

# 输出映射字典
print("Mapping Dictionary:")
for key, value in mapping_dict.items():
    print(f"{key}: {value}")

# 输出字符及其对应的独热编码向量
print("\nCharacter and Its One-Hot Encoding:")
for character in pos_tags:
    onehot_encoding = mapping_dict.get(character)
    print(f"Character: {character}, One-Hot Encoding: {onehot_encoding}")
print('char onehot over')
#-----------------------------------------------------------------------------------------------

# 指定文件夹路径
folder_path = "D:/24知识工程/第一次编程作业/20180712165812468713/04-现代汉语切分、标注、注音语料库-1998年1月份样例与规范20110330"
folder_path_json = "D:/24知识工程/第一次编程作业/coref-dataset/coref-dataset/train"
# 创建字典用于记录 ID 值和行内容及索引信息
id_lines_dict = {}

# 检查指定文件夹下的所有文件，获取 id_lines_dict
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='gbk') as file:
            content = file.readlines()
            for file_name_json in os.listdir(folder_path_json):
                if file_name_json.endswith(".json"):
                    file_path_json = os.path.join(folder_path_json, file_name_json)
                    with open(file_path_json, 'r', encoding="gbk") as file_json:
                        label = json.load(file_json)
                        if label:
                            task_id = label.get("taskID", "未找到")
                            value_0 = label.get("0", {}).get("id", "未找到")
                            index_front = label.get("0", {}).get("indexFront", "未找到")
                            index_behind = label.get("0", {}).get("indexBehind", "未找到")
                            pronoun_index_front = label.get("pronoun", {}).get("indexFront", "未找到")
                            pronoun_index_behind = label.get("pronoun", {}).get("indexBehind", "未找到")

                            for line in content:
                                if value_0 in line:
                                    if task_id not in id_lines_dict:
                                        id_lines_dict[task_id] = {}
                                    id_lines_dict[task_id][value_0] = {
                                        "line_content": line.strip(),
                                        "0_index_front": index_front,
                                        "0_index_behind": index_behind,
                                        "pronoun_index_front": pronoun_index_front,
                                        "pronoun_index_behind": pronoun_index_behind
                                    }
result = []

# 遍历字典中的每个 ID 和行内容
for task_id, value_0_dict in tqdm(id_lines_dict.items(), desc='Processing tasks'):
    # 在此处执行外层循环体操作
    for value_0, line_dict in value_0_dict.items():

        line = line_dict['line_content']

        # 找到第一个汉字的索引
        first_chinese_index = next((i for i, char in enumerate(line) if '\u4e00' <= char <= '\u9fff'), None)

        # 找到第一个汉字后的斜杠索引
        slash_indices = [i for i, char in enumerate(line[first_chinese_index:]) if char == "/"]

        # 遍历斜杠索引
        for start_index in slash_indices:
            substring = ""
            new_substring = ""
            index = first_chinese_index + start_index + 1
            i = 0  # 在进入一句换的遍历的开始阶段将i初始化为0，这时候i指向的是第0个字
            if '\u4e00' < line[index - 2] < '\u9fff':

                # 跳过第一个斜杠后的字符
                if index < len(line):
                    # 遍历直到遇到空格或结尾
                    while index < len(line) and line[index] != " ":
                        substring += line[index]
                        index += 1

                    # 检查子字符串是否在映射字典中
                    if substring != "":
                        onehot_encoding = mapping_dict.get(substring)
                        result.append((onehot_encoding))  # 这里substring不要加入特征表，只存编码就可以
                    # 在这里添加下一步的代码，计算其他特征值
                    #
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

                                    result.append(onehot_encoding + new_onehot_encoding)



                            i = i + 1






        # 打印结果
        print("分析结果:")
        for item in result:
            print(item)
            tqdm.write(f"Progress: {i}/{len(value_0_dict)}")  # 更新进度条位置
        print('over!!!!')
