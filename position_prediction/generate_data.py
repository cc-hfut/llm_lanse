"""Generate reproducible data files for the position-prediction-error experiment."""
from __future__ import annotations
import csv, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from _data import load_kg,split_triples

MODELS=['TransE','TransERR','QuatE','QuatE-D','RotatE','RoCS','TFKGE','SR-KGE','LLM-GeoSE']
def main():
    if not (ROOT/'data/landslide.txt').exists():
        raise FileNotFoundError('data/landslide.txt is private and is not distributed. Supply your own KG input before generating position-prediction data. See README.md.')
    data=load_kg(ROOT/'data/landslide.txt',ROOT/'data/nasa_global_landslide_catalog_point.csv')
    _,_,test=split_triples(data.triples,seed=2025); out=Path(__file__).parent/'data'; out.mkdir(exist_ok=True)
    with open(out/'entity_coordinates.csv','w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['entity_id','entity_name','latitude','longitude'])
        for i,(lat,lon) in sorted(data.coordinates.items()): w.writerow([i,data.id2entity[i],lat,lon])
    rid=data.relation2id['gazetteer_closest_point']; spatial=[x for x in test if x[1]==rid and x[0] in data.coordinates]
    with open(out/'spatial_test.csv','w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['query_id','head_id','head_name','relation_id','relation','tail_id','tail_name','true_location_entity_id','evaluation_side'])
        for q,(h,r,t) in enumerate(spatial): w.writerow([q,h,data.id2entity[h],r,data.id2relation[r],t,data.id2entity[t],h,'head'])
    with open(out/'candidate_entities.csv','w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['entity_id','entity_name'])
        for i in sorted(data.coordinates): w.writerow([i,data.id2entity[i]])
    pred=out/'predictions'; pred.mkdir(exist_ok=True)
    for model in MODELS:
        target=pred/f'{model}.csv'
        # Rebuilding evaluation inputs must not erase completed model predictions.
        if target.exists():
            continue
        with open(target,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(['model','query_id','true_entity','predicted_entity','relation'])
            for q,(h,r,t) in enumerate(spatial): w.writerow([model,q,data.id2entity[h],'',data.id2relation[r]])
    print({'coordinate_entities':len(data.coordinates),'spatial_test':len(spatial),'candidate_entities':len(data.coordinates),'prediction_templates':len(MODELS)})
if __name__=='__main__': main()
