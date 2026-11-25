# -*- coding: utf-8 -*-
# @Time    : 2025/6/19 22:00
# @Author  : cc
# @File    : entity2id.py
# @Description: Map entities and relations to IDs.
import json

# Open the triple file for reading with UTF-8 encoding.
with open('landslide.txt', 'r', encoding='utf-8') as f:
    # Collect all head and tail entities.
    entity = []
    # Collect relation names.
    rels = []

    # Read each triple line.
    for line in f:
        # Strip whitespace and split tab-separated h, r, and t fields.
        h, r, t = line.strip().split('\t')
        # Append the head entity.
        entity.append(str(h))
        # Append the tail entity.
        entity.append(str(t))
        # Append the relation.
        rels.append(str(r))

    # Deduplicate entity names.
    entity = list(set(entity))
    # Deduplicate relation names.
    rels = list(set(rels))

    # Create the entity-to-ID mapping.
    entity2id_dict = {}
    # Open entity2id.txt for writing entity IDs.
    entity2id = open('entity2id.txt', 'w', encoding='utf-8')
    # Open rel2id.txt for writing relation IDs.
    rel2id = open('rel2id.txt', 'w', encoding='utf-8')
    entitys = open('entitys.txt', 'w', encoding='utf-8')
    # Write each entity and its unique ID, separated by a tab.
    for i in range(len(entity)):
        entity2id_dict[entity[i]] = i
        entity2id.write(entity[i] + '\t' + str(i) + '\n')
        entitys.write(entity[i]+ '\n')

    # Write each relation and its unique ID, separated by a tab.
    for i in range(len(rels)):
        rel2id.write(rels[i] + '\t' + str(i) + '\n')
    
    # Close the files.
    entity2id.close()
    rel2id.close()
    entitys.close()

