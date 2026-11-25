'Create reproducible dataset splits while retaining the original triple TXT format.'
from pathlib import Path
import random

SEED = 2025
DATA = Path.cwd()
source = DATA / "landslide.txt"
rows = [line.rstrip("\n") for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
for line_no, row in enumerate(rows, 1):
    if len(row.split("\t")) != 3:
        raise ValueError(f"landslide.txt line {line_no} is not a valid triple")
random.Random(SEED).shuffle(rows)
n_train = int(len(rows) * 0.90)
n_valid = int(len(rows) * 0.09)
parts = {
    "train.txt": rows[:n_train],
    "valid.txt": rows[n_train:n_train + n_valid],
    "test.txt": rows[n_train + n_valid:],
}
for name, values in parts.items():
    (DATA / name).write_text("".join(x + "\n" for x in values), encoding="utf-8")
print({name: len(values) for name, values in parts.items()})
