from __future__ import annotations
import math, random


def haversine_km(a,b):
    lat1,lon1=map(math.radians,a); lat2,lon2=map(math.radians,b)
    x=math.sin((lat2-lat1)/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2*6371.0088*math.asin(min(1.,math.sqrt(x)))


def geo_metrics(records, coordinates, seed=2025, random_repeats=1000):
    """records: iterable of (true_entity, predicted_entity, relation, candidate_ids)."""
    rows=[]
    for true,pred,rel,candidates in records:
        if true in coordinates and pred in coordinates:
            pool=[x for x in candidates if x in coordinates]
            if pool: rows.append((true,pred,rel,pool))
    if not rows: raise ValueError("No evaluable records with coordinates")
    errors=[haversine_km(coordinates[t],coordinates[p]) for t,p,_,_ in rows]
    ordered=sorted(errors); n=len(errors)
    rng=random.Random(seed); random_mpes=[]
    for _ in range(random_repeats):
        random_mpes.append(sum(haversine_km(coordinates[t],coordinates[rng.choice(pool)]) for t,_,_,pool in rows)/n)
    return {"n":n,"mpe_km":sum(errors)/n,"median_km":ordered[n//2],"acc_50":sum(x<=50 for x in errors)/n,
            "acc_100":sum(x<=100 for x in errors)/n,"acc_500":sum(x<=500 for x in errors)/n,
            "random_mpe_km":sum(random_mpes)/len(random_mpes)}
