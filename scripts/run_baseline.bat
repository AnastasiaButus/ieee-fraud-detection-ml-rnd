@echo off
REM Выполнение из корня проекта в Anaconda Prompt (Windows).
python -m src.train --config configs/baseline.json || exit /b 1
python -m src.evaluate --config configs/baseline.json || exit /b 1
