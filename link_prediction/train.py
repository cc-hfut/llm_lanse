from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import torch
from common.data import load_kg, split_triples
from common.models import TransE,TransERR,QuatE,QuatED,RotatE,RoCS,TFKGE,QiuSTCM,LLMGeoSE
from common.geo_eval import geo_metrics


def tensors(data,device): return tuple(torch.tensor(x,device=device) for x in zip(*data))
def metadata(data,device):
    tm=torch.zeros((len(data.id2entity),6),dtype=torch.float32,device=device)
    cm=torch.zeros((len(data.id2entity),2),dtype=torch.float32,device=device)
    for i,x in data.times.items(): tm[i]=torch.tensor(x,device=device)
    for i,x in data.coordinates.items(): cm[i]=torch.tensor(x,device=device)
    return tm,cm
def negatives(batch,nentity,k,rng):
    out=[]
    for h,r,t in batch: out.append([(h,r,rng.randrange(nentity)) for _ in range(k)])
    return [x for row in out for x in row]
def rank_tail(model,triple,nentity,known,device):
    h,r,t=triple; ids=torch.arange(nentity,device=device); hh=torch.full_like(ids,h); rr=torch.full_like(ids,r)
    with torch.no_grad(): scores=model(hh,rr,ids)
    fill=-torch.inf if model.higher_is_better else torch.inf
    for cand in known.get((h,r),()):
        if cand!=t: scores[cand]=fill
    order=torch.argsort(scores,descending=model.higher_is_better); rank=int((order==t).nonzero()[0])+1
    return rank,int(order[0])
def rank_head(model,triple,nentity,known,device,candidate_ids=None):
    h,r,t=triple; ids=torch.arange(nentity,device=device); rr=torch.full_like(ids,r); tt=torch.full_like(ids,t)
    with torch.no_grad(): scores=model(ids,rr,tt)
    fill=-torch.inf if model.higher_is_better else torch.inf
    if candidate_ids is not None:
        allowed=torch.zeros(nentity,dtype=torch.bool,device=device)
        allowed[torch.tensor(candidate_ids,device=device)]=True
        scores[~allowed]=fill
    for cand in known.get((r,t),()):
        if cand!=h: scores[cand]=fill
    order=torch.argsort(scores,descending=model.higher_is_better); return int(order[0])


def main_for(fixed_model=None):
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['transe','transerr','quate','quate-d','rotate','rocs','tfkge','qiu-stcm','llm-geose'],required=fixed_model is None)
    default_data=str(Path(__file__).resolve().parents[1]/'data')
    default_output=str(Path(__file__).resolve().parent/'training_results')
    p.add_argument('--data-dir',default=default_data); p.add_argument('--dim',type=int,default=200); p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=512); p.add_argument('--negatives',type=int,default=10); p.add_argument('--seed',type=int,default=2025)
    p.add_argument('--eval-every',type=int,default=10); p.add_argument('--patience',type=int,default=5); p.add_argument('--valid-sample',type=int,default=256)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--output-dir',default=default_output)
    a=p.parse_args(); a.model=fixed_model or a.model; random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device(a.device)
    root=Path(a.data_dir); data=load_kg(root/'landslide.txt',root/'nasa_global_landslide_catalog_point.csv')
    train,valid,test=split_triples(data.triples,a.seed); tm,cm=metadata(data,device); ne,nr=len(data.id2entity),len(data.id2relation)
    factories={'transe':lambda:TransE(ne,nr,a.dim),'transerr':lambda:TransERR(ne,nr,a.dim),'quate':lambda:QuatE(ne,nr,a.dim),
        'quate-d':lambda:QuatED(ne,nr,a.dim),'rotate':lambda:RotatE(ne,nr,a.dim),'rocs':lambda:RoCS(ne,nr,a.dim),
        'tfkge':lambda:TFKGE(ne,nr,tm,a.dim),'qiu-stcm':lambda:QiuSTCM(ne,nr,tm,cm,a.dim),
        'llm-geose':lambda:LLMGeoSE(ne,nr,a.dim)}
    model=factories[a.model]().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); rng=random.Random(a.seed)
    best_mrr=-1.; stale=0; best_state=None
    for epoch in range(a.epochs):
        rng.shuffle(train); model.train(); total=0.
        for i in range(0,len(train),a.batch_size):
            batch=train[i:i+a.batch_size]; neg=negatives(batch,ne,a.negatives,rng)
            ps=model(*tensors(batch,device)); ns=model(*tensors(neg,device)).view(len(batch),a.negatives)
            loss=model.loss(ps,ns); opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss.detach())*len(batch)
        if epoch%a.eval_every==0 or epoch==a.epochs-1:
            known={}
            for h,r,t in train+valid+test: known.setdefault((h,r),set()).add(t)
            sample=valid[:min(a.valid_sample,len(valid))]; vr=[rank_tail(model,x,ne,known,device)[0] for x in sample]
            vmrr=sum(1/x for x in vr)/len(vr); print(f'epoch={epoch} loss={total/len(train):.6f} valid_mrr={vmrr:.6f}')
            if vmrr>best_mrr+1e-6:
                best_mrr=vmrr; stale=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale+=1
                if stale>=a.patience: print(f'early_stop epoch={epoch}'); break
    if best_state is not None: model.load_state_dict(best_state)
    known_tail={}; known_head={}
    for h,r,t in train+valid+test: known_tail.setdefault((h,r),set()).add(t); known_head.setdefault((r,t),set()).add(h)
    ranks=[]; predictions=[]
    model.eval()
    for tr in test:
        rank,pred=rank_tail(model,tr,ne,known_tail,device); ranks.append(rank); predictions.append((tr,pred))
    result={'model':a.model,'protocol':'deterministic 90/9/1 split; filtered tail ranking','seed':a.seed,'best_valid_mrr':best_mrr,
            'n_test':len(test),'hits@1':sum(x<=1 for x in ranks)/len(ranks),'hits@3':sum(x<=3 for x in ranks)/len(ranks),
            'hits@10':sum(x<=10 for x in ranks)/len(ranks),'mr':sum(ranks)/len(ranks),'mrr':sum(1/x for x in ranks)/len(ranks)}
    # Current KG has coordinates on landslide heads, not gazetteer tails: evaluate filtered head prediction.
    spatial_r=data.relation2id.get('gazetteer_closest_point'); geo=[]; coord_candidates=sorted(data.coordinates)
    geo_predictions=[]
    if spatial_r is not None:
        for tr in test:
            if tr[1]==spatial_r and tr[0] in data.coordinates:
                pred=rank_head(model,tr,ne,known_head,device,coord_candidates); geo.append((tr[0],pred,tr[1],coord_candidates)); geo_predictions.append((tr,pred))
    if geo: result['geo_head_prediction']=geo_metrics(geo,data.coordinates,a.seed)
    display={'transe':'TransE','transerr':'TransERR','quate':'QuatE','quate-d':'QuatE-D','rotate':'RotatE',
             'rocs':'RoCS','tfkge':'TFKGE','qiu-stcm':'SR-KGE','llm-geose':'LLM-GeoSE'}[a.model]
    out=Path(a.output_dir)/display; out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.save({'state_dict':model.state_dict(),'entity2id':data.entity2id,'relation2id':data.relation2id},out/f'best_{display}.pt')
    # Formal Top-1 output consumed directly by the position-prediction-error experiment.
    pred_dir=Path(__file__).resolve().parents[1]/'position_prediction'/'data'/'predictions'; pred_dir.mkdir(parents=True,exist_ok=True)
    rows=[[display,q,data.id2entity[tr[0]],data.id2entity[pred],data.id2relation[tr[1]]] for q,(tr,pred) in enumerate(geo_predictions)]
    for path in (pred_dir/f'{display}.csv',out/'top1.csv'):
        with open(path,'w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f); w.writerow(['model','query_id','true_entity','predicted_entity','relation']); w.writerows(rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))
def main(): main_for()
if __name__=='__main__': main()
