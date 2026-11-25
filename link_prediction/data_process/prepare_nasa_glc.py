'Prepare experiment inputs from NASA GLC and the separately supplied private KG.'
from __future__ import annotations
import csv, io, json, random, re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
SEED = 2025

def event_name(row):
    country=(row.get('country_name') or 'unknown').strip().replace(' ','_')
    admin=(row.get('admin_division_name') or 'unknown').strip().replace(' ','_')
    return f'{country}_{admin}_landslide'

def read_catalog(path):
    try: raw=path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError: raw=path.read_text(encoding='latin-1')
    return csv.DictReader(io.StringIO(raw,newline=''))

def relation_type(name):
    # Escaped Chinese labels match relations in private KG and landslide inputs.
    # Preserve their values for compatibility with the original data schema.
    s=name.lower()
    if any(x in s for x in ('time','date','\u65f6\u95f4')): return 1
    if any(x in s for x in ('location','gazetteer','spatial','\u7a7a\u95f4','\u4f4d\u4e8e','\u90bb\u8fd1','\u4e0a\u6e38','\u4e0b\u6e38')): return 2
    if any(x in s for x in ('trigger','if-then','\u5f71\u54cd','\u7834\u574f','\u5a01\u80c1','\u5185\u56e0','\u5916\u56e0')): return 3
    return 0

def main():
    triple_path=DATA/'landslide.txt'; catalog_path=DATA/'nasa_global_landslide_catalog_point.csv'
    if not triple_path.exists():
        raise FileNotFoundError('data/landslide.txt is private and is not distributed. Supply your own KG input; the NASA CSV alone cannot reconstruct it. See README.md.')
    if not catalog_path.exists():
        raise FileNotFoundError('Missing data/nasa_global_landslide_catalog_point.csv')
    rows=[]; entities=set(); relations=set(); adjacency=defaultdict(list)
    for no,line in enumerate(triple_path.read_text(encoding='utf-8').splitlines(),1):
        p=line.split('\t')
        if len(p)!=3: raise ValueError(f'landslide.txt line {no} is not a valid triple')
        h,r,t=p; rows.append((h,r,t)); entities.update((h,t)); relations.add(r); adjacency[h].extend((r,t)); adjacency[t].append(r)
    id2entity=sorted(entities); id2relation=sorted(relations)
    ent2id={x:i for i,x in enumerate(id2entity)}; rel2id={x:i for i,x in enumerate(id2relation)}
    descriptions=defaultdict(list); times=defaultdict(list); locations=defaultdict(list); coords=defaultdict(list)
    for row in read_catalog(catalog_path):
        name=event_name(row)
        if name not in ent2id: continue
        desc='；'.join(x.strip() for x in (row.get('event_title',''),row.get('event_description',''),row.get('location_description','')) if x and x.strip())
        if desc: descriptions[name].append(desc)
        if row.get('event_date'): times[name].append(row['event_date'].strip())
        loc=row.get('location_description') or row.get('gazetteer_closest_point')
        if loc: locations[name].append(loc.strip())
        try:
            lat,lon=float(row['latitude']),float(row['longitude'])
            if -90<=lat<=90 and -180<=lon<=180: coords[name].append((lat,lon))
        except (ValueError,TypeError): pass
    def first_or(values,default): return values[0] if values else default
    # Escaped fallback text is retained for private KG/landslide preprocessing.
    # Keeping the original values avoids changing downstream embedding inputs.
    entityid2duc={str(i):first_or(descriptions[name],f'\u5730\u7406\u77e5\u8bc6\u56fe\u8c31\u5b9e\u4f53\uff1a{name}') for i,name in enumerate(id2entity)}
    id2time={str(i):first_or(times[name],'\u65e0\u65f6\u95f4') for i,name in enumerate(id2entity)}
    id2location={str(i):first_or(locations[name],'\u65e0\u5750\u6807') for i,name in enumerate(id2entity)}
    entity2time={name:first_or(times[name],'\u65e0\u65f6\u95f4') for name in id2entity}
    entity2location={name:first_or(locations[name],'\u65e0\u5750\u6807') for name in id2entity}
    keywords={str(i):list(dict.fromkeys([name]+[str(x) for x in adjacency[name]])) for i,name in enumerate(id2entity)}
    reltypes={str(i):relation_type(name) for i,name in enumerate(id2relation)}
    coordinates=[]
    for name,values in coords.items():
        coordinates.append((ent2id[name],name,sum(x for x,_ in values)/len(values),sum(y for _,y in values)/len(values)))
    (DATA/'entity2id.txt').write_text(''.join(f'{x}\t{i}\n' for i,x in enumerate(id2entity)),encoding='utf-8')
    (DATA/'rel2id.txt').write_text(''.join(f'{x}\t{i}\n' for i,x in enumerate(id2relation)),encoding='utf-8')
    (DATA/'entitys.txt').write_text(''.join(f'{x}\n' for x in id2entity),encoding='utf-8')
    for name,obj in [('entityid2duc.json',entityid2duc),('id2time.json',id2time),('id2location.json',id2location),('entity2time.json',entity2time),('entity2location.json',entity2location),('entityid2keywords.json',keywords),('relid2typeid.json',reltypes)]:
        (DATA/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    with open(DATA/'entity_coordinates.csv','w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f); w.writerow(['entity_id','entity_name','latitude','longitude']); w.writerows(coordinates)
    shuffled=list(rows); random.Random(SEED).shuffle(shuffled); a=int(len(shuffled)*.9); b=int(len(shuffled)*.99)
    for name,part in [('train.txt',shuffled[:a]),('valid.txt',shuffled[a:b]),('test.txt',shuffled[b:])]:
        (DATA/name).write_text(''.join('\t'.join(x)+'\n' for x in part),encoding='utf-8')
    manifest={'seed':SEED,'entities':len(id2entity),'relations':len(id2relation),'triples':len(rows),'train':a,'valid':b-a,'test':len(rows)-b,'coordinate_entities':len(coordinates)}
    (DATA/'experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
