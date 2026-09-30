@echo off
rem Starts the app and prints a public link you can send to a friend. Close this window to stop sharing.
cd /d "%~dp0"
.venv\Scripts\python share.py
pause
