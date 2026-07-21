@echo off
REM Tum NGC/IC gorsel referans indeksini indirir (kaldigi yerden devam eder)
REM ve embeddingleri gunceller. Bitene kadar acik birakin (~1-2 saat).
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\build_visual_index.py
.venv\Scripts\python.exe scripts\build_visual_embeddings.py
echo.
echo BITTI - pencereyi kapatabilirsiniz.
pause
