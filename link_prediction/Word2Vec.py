# -*- coding: utf-8 -*-
# @Author  : cc
# @File    : entity_encoder.py
# @Description: Pretrain Word2Vec on keywords.
from gensim.models import Word2Vec
import json

only_keywords = False
with open('data\\entityid2keywords.json', 'r', encoding='utf-8') as f:
    entityid2keywords = json.load(f)  # Parse a Python dictionary or list.
# Build a corpus with one token sequence per entity.
sentences = list(entityid2keywords.values())


if not only_keywords:
    with open('data\\entity2id.txt', 'r', encoding='utf-8') as f:
        words = []
        lines = f.readlines()
        for i in range(len(lines)):
            words.append(lines[i].strip().split('\t')[0])
        sentences.append(words)
    with open('data\\rel2id.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        words = []
        for i in range(len(lines)):
            words.append(lines[i].strip().split('\t')[0])
        sentences.append(words)
# words = list(set(words))
# if 'specific' in words:
#     print('specific')
# Word2Vec configuration
model = Word2Vec(
    sentences=sentences,
    vector_size=100,  # Embedding dimension
    window=3,  # Context window
    min_count=1,  # Minimum token count
    sg=1,  # 1: skip-gram; 0: CBOW
    workers=4,
    epochs=10000
)

# Extract KeyedVectors.
word_vectors = model.wv

# Save the vectors.
if only_keywords:
    word_vectors.save("word2vec_keywords.kv")
else:
    word_vectors.save("word2vec_all_words.kv")
