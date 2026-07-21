@echo off
cd /d "%~dp0"
echo Starting Celestial Object Analyzer at http://127.0.0.1:8777 ...
start "" http://127.0.0.1:8777
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8777
