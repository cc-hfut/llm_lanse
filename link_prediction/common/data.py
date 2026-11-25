from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class KGData:
    triples: list[tuple[int, int, int]]
    entity2id: dict[str, int]
    relation2id: dict[str, int]
    id2entity: list[str]
    id2relation: list[str]
    coordinates: dict[int, tuple[float, float]]
    times: dict[int, tuple[float, float, float, float, float, float]]


def _event_name(row: dict[str, str]) -> str:
    country = (row.get("country_name") or "unknown").strip().replace(" ", "_")
    admin = (row.get("admin_division_name") or "unknown").strip().replace(" ", "_")
    return f"{country}_{admin}_landslide"


def _parse_time(value: str):
    if not value:
        return None
    value = value.strip().split(" ")[0]
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return (float(dt.year), float(dt.month), float(dt.day), 0.0, 0.0, 0.0)
        except ValueError:
            pass
    return None


def load_kg(triple_path: str | Path, catalog_path: str | Path) -> KGData:
    named = []
    entities, relations = set(), set()
    with open(triple_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            named.append((h, r, t)); entities.update((h, t)); relations.add(r)
    id2entity = sorted(entities); id2relation = sorted(relations)
    entity2id = {x: i for i, x in enumerate(id2entity)}
    relation2id = {x: i for i, x in enumerate(id2relation)}
    triples = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in named]

    # The cleaned KG collapses events sharing country/admin into one entity. Aggregate
    # their coordinates/time instead of silently keeping an arbitrary catalog row.
    coord_acc: dict[int, list[tuple[float, float]]] = {}
    time_acc: dict[int, list[tuple[float, ...]]] = {}
    # The supplied ArcGIS export is Windows-1252 although its header is ASCII.
    # latin-1 is a lossless fallback for entity matching because the generated
    # event key uses country/admin fields, which are ASCII in this snapshot.
    try:
        raw = Path(catalog_path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = Path(catalog_path).read_text(encoding="latin-1")
    import io
    with io.StringIO(raw, newline="") as f:
        for row in csv.DictReader(f):
            eid = entity2id.get(_event_name(row))
            if eid is None:
                continue
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    coord_acc.setdefault(eid, []).append((lat, lon))
            except (TypeError, ValueError):
                pass
            tm = _parse_time(row.get("event_date", ""))
            if tm:
                time_acc.setdefault(eid, []).append(tm)
    coordinates = {k: (sum(x[0] for x in v)/len(v), sum(x[1] for x in v)/len(v)) for k, v in coord_acc.items()}
    times = {k: tuple(sum(x[j] for x in v)/len(v) for j in range(6)) for k, v in time_acc.items()}
    return KGData(triples, entity2id, relation2id, id2entity, id2relation, coordinates, times)


def split_triples(triples, seed=2025, train_ratio=.90, valid_ratio=.09):
    items = list(triples); random.Random(seed).shuffle(items)
    a = int(len(items) * train_ratio); b = int(len(items) * (train_ratio + valid_ratio))
    return items[:a], items[a:b], items[b:]
