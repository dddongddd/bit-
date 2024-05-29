import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import re
from tqdm import tqdm
import pandas as pd

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

print (id_lines_dict)

