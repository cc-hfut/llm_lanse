#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python LLM-GeoSE.py --ablation wo_relation_enhancement
