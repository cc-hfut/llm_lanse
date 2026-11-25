# LLM-GeoSE

## Project structure

| Directory | Purpose | Main entry points |
| --- | --- | --- |
| `link_prediction/` | Link prediction with LLM-GeoSE and baselines | `LLM-GeoSE.py` and model-named scripts such as `TransE.py` |
| `position_prediction/` | Geographic position error evaluation | `Random.py` and model-named scripts |
| `ablation/` | Seven LLM-GeoSE ablation variants | `LLM-GeoSE.py` and `01_*.sh` through `07_*.sh` |

## Data and model

Only the public NASA catalog, `data/nasa_global_landslide_catalog_point.csv`, is included. The private knowledge graph (`landslide.txt`), derived experiment inputs, and Word2Vec weights cannot be shared. Full experiments require these inputs; the public CSV alone is insufficient to reproduce the reported results. Unicode-escaped labels in preprocessing scripts preserve compatibility with private KG and landslide data.

Download the embedding model from [Qwen/Qwen3-Embedding-8B-GGUF on Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF):

```powershell
python -m pip install -r requirements.txt
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3-Embedding-8B-GGUF Qwen3-Embedding-8B-f16.gguf --local-dir Qwen3-Embedding-8B-GGUF
```

Alternatively, set `QWEN3_EMBEDDING_MODEL` to the local GGUF path. This embedding model differs from Qwen3-MAX used for description generation; the latter's weights and generation service are not bundled.

## Usage

After supplying the required experiment inputs, run each entry point from its own directory:

```powershell
cd link_prediction
python LLM-GeoSE.py --ablation full
python TransE.py
cd ../position_prediction
python Random.py
python LLM-GeoSE.py
```

Position evaluation reads predictions exported by the corresponding link-prediction model. For ablations, use Git Bash or WSL:

```bash
cd ablation
bash 01_wo_entity_enhancement.sh
# Run the other 02_*.sh through 07_*.sh scripts in the same way.
```

Training outputs: `training_results/` within each training directory. Position metrics: `position_prediction/results/`.

## Acknowledgments

TFKGE components reference [Date2Vec](https://github.com/ojus1/Date2Vec), [SE-KGE](https://github.com/gengchenmai/se-kge), [graphqembed](https://github.com/williamleif/graphqembed), and [MMKB](https://github.com/mniepert/mmkb).
