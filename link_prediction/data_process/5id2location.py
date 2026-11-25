import json

with open('entity2location.json', 'r', encoding='utf-8') as f:
    entity2location = json.load(f)
with open('entity2time.json', 'r', encoding='utf-8') as f:
    entity2time = json.load(f)
entity2id = {}
with open('entity2id.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        name, idx = parts
        entity2id[name] = int(idx)
id2location = {}
id2time = {}
# Escaped placeholders preserve the private KG/landslide preprocessing format.
for name, id in entity2id.items():
    if name in entity2location:
        id2location[id] = entity2location[name]
    else:
        id2location[id] = '\u65e0\u5750\u6807'
    if name in entity2time:
        id2time[id] = entity2time[name]
    else:
        id2time[id] = '\u65e0\u65f6\u95f4'

with open('id2location.json', 'w', encoding='utf-8') as f:
    json.dump(id2location, f, ensure_ascii=False, indent=4)
with open('id2time.json', 'w', encoding='utf-8') as f:
    json.dump(id2time, f, ensure_ascii=False, indent=4)
