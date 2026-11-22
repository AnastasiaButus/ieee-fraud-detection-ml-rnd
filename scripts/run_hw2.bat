@echo off
REM Запуск из корневого каталога проекта ДЗ1 в Anaconda Prompt.
python -m src.hw2_experiments --config configs/hw2_experiments.json || exit /b 1
