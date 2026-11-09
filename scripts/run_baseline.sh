#!/usr/bin/env bash
# Выполнение из корня проекта в Unix-подобной среде.
set -euo pipefail
python -m src.train --config configs/baseline.json
python -m src.evaluate --config configs/baseline.json
