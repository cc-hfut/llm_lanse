# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 22:56
# @Author  : cc
# @File    : entityid2keywords.py
# @Description: Map entity IDs to their descriptions.
import pandas as pd
import json
import numpy as np

# Read the input file.
excel_file = pd.ExcelFile('output.xlsx')

# List worksheet names.
sheet_names = excel_file.sheet_names

# Read data from the selected worksheet.
df = excel_file.parse('Sheet1')
df = df.replace({np.nan: 'None'})
# Escaped Chinese column labels belong to the private KG/landslide worksheet.
# They preserve matching with the original input schema; no private data is bundled.
entity = df['\u5b9e\u4f53\u540d\u79f0']
ducs = df['\u5b9e\u4f53\u63cf\u8ff0']
entity2id = {}
with open('entity2id.txt', 'r', encoding='utf-8') as f:
    for line in f:
        name, idx = line.strip().split('\t')  # Read the name and its ID.
        entity2id[str(name)] = int(idx)  # Convert the ID to an integer.

entityid2duc = {}
for name, id in entity2id.items():
    try:
        index = df[df['\u5b9e\u4f53\u540d\u79f0'] == name].index[0]
        entityid2duc[id] = ducs[index]
    except:
        # Preserve the original private-data placeholder used during preprocessing.
        entityid2duc[id] = '\u65e0\u63cf\u8ff0\u6587\u672c'
with open('entityid2duc.json', 'w', encoding='utf-8') as f:
    json.dump(entityid2duc, f, ensure_ascii=False, indent=4)
