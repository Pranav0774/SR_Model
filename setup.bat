@echo off
rem One-time setup for Windows. Needs Python 3.10-3.12 and ffmpeg on PATH.
cd /d "%~dp0"

where ffmpeg >nul 2>nul || (echo ffmpeg was not found. Install it from https://ffmpeg.org and add it to PATH, then run this again. & exit /b 1)

if not exist .venv (
    py -3.11 -m venv .venv || py -3.12 -m venv .venv || py -3.10 -m venv .venv || (echo Could not create a virtual environment. Install Python 3.11 first. & exit /b 1)
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
.venv\Scripts\python download_models.py
