from __future__ import annotations
import csv, json
from pathlib import Path
import torch

def save_metrics(result_dir, model_name, hits, mr, mrr):
    out=Path(result_dir); out.mkdir(parents=True,exist_ok=True)
    obj={'model':model_name,'Hits@1':hits[1],'Hits@3':hits[3],'Hits@10':hits[10],
         'MR':float(mr),'MRR':float(mrr)}
    (out/'metrics.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def export_geo_head_top1(model, model_name, test_data, all_true_triples, ent2id, rel2id,
                         result_dir, device, chunk_size=256):
    root=Path(__file__).resolve().parents[2]
    coords=set()
    with open(root/'data'/'entity_coordinates.csv',encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f): coords.add(int(row['entity_id']))
    rid=rel2id.get('gazetteer_closest_point'); id2ent={v:k for k,v in ent2id.items()}; id2rel={v:k for k,v in rel2id.items()}
    candidates=sorted(coords); records=[]; model.eval()
    with torch.no_grad():
        for h,r,t in test_data:
            if r!=rid or h not in coords: continue
            values=[]
            for start in range(0,len(candidates),chunk_size):
                part=candidates[start:start+chunk_size]; hh=torch.tensor(part,dtype=torch.long,device=device)
                values.append(model(hh,torch.full_like(hh,r),torch.full_like(hh,t)).detach().cpu())
            scores=torch.cat(values)
            for i,cand in enumerate(candidates):
                if cand!=h and (cand,r,t) in all_true_triples: scores[i]=float('inf')
            pred=candidates[int(torch.argmin(scores))]
            records.append((model_name,len(records),id2ent[h],id2ent[pred],id2rel[r]))
    out=Path(result_dir); targets=[out/'top1.csv',root/'position_prediction'/'data'/'predictions'/f'{model_name}.csv']
    for path in targets:
        path.parent.mkdir(parents=True,exist_ok=True)
        with open(path,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(['model','query_id','true_entity','predicted_entity','relation']); w.writerows(records)
