from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from _geo_eval import geo_metrics

HERE=Path(__file__).parent; DATA=HERE/'data'
def load():
    coords={}; names={}
    with open(DATA/'entity_coordinates.csv',encoding='utf-8-sig',newline='') as f:
        for x in csv.DictReader(f): coords[int(x['entity_id'])]=(float(x['latitude']),float(x['longitude'])); names[x['entity_name']]=int(x['entity_id'])
    pool=sorted(coords); return coords,names,pool
def evaluate_model(model):
    coords,names,pool=load(); path=DATA/'predictions'/f'{model}.csv'; rows=[]
    with open(path,encoding='utf-8-sig',newline='') as f:
        for x in csv.DictReader(f):
            if not x['predicted_entity'].strip(): raise ValueError(f'{path}: query {x["query_id"]} has no predicted_entity')
            rows.append((names[x['true_entity']],names[x['predicted_entity']],x['relation'],pool))
    result=geo_metrics(rows,coords,seed=2025,random_repeats=1000); out=HERE/'results'; out.mkdir(exist_ok=True)
    (out/f'{model}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
def evaluate_random():
    coords,names,pool=load(); rng=random.Random(2025); truths=[]
    with open(DATA/'spatial_test.csv',encoding='utf-8-sig',newline='') as f:
        for x in csv.DictReader(f): truths.append(int(x['true_location_entity_id']))
    records=[(t,rng.choice(pool),'gazetteer_closest_point',pool) for t in truths]
    result=geo_metrics(records,coords,seed=2025,random_repeats=1000); result={'n':result['n'],'mpe_km':result['random_mpe_km']}
    out=HERE/'results'; out.mkdir(exist_ok=True); (out/'Random.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
