#!/usr/bin/env bash
# Запуск из корневого каталога проекта.
set -euo pipefail
python -m src.hw2_experiments --config configs/hw2_experiments.json
