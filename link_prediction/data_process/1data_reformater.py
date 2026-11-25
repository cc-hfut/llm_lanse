# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 23:22
# @Author  : cc
# @File    : data_reformater.py
# @Description: Remove description relations from the triple input.
import json

triple = []
entity2time = {}
entity2location = {}

with open('landslideKG.txt', 'r', encoding='utf-8') as f:
    # Read each triple line.
    for line in f:
        # Strip whitespace and split tab-separated h, r, and t fields.
        h, r, t = line.strip().split('\t')
        if r != 'event_description' and r != 'location_description' and r != 'event_date' and r != 'location':
            triple.append((h, r, t))
        if r == 'event_date':
            entity2time[h] = t
        if r == 'location':
            entity2location[h] = t

with open('landslide.txt', 'w', encoding='utf-8') as o:
    for row in triple:
        o.write('\t'.join(map(str, row)) + '\n')
with open('entity2time.json', 'w', encoding='utf-8') as f:
    json.dump(entity2time, f, ensure_ascii=False, indent=4)
with open('entity2location.json', 'w', encoding='utf-8') as f:
    json.dump(entity2location, f, ensure_ascii=False, indent=4)
