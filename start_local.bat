@echo off
REM Runs the transcriber on THIS computer only (no remote link, no password).
cd /d "%~dp0"
echo Starting the transcriber... your browser will open shortly.
.venv\Scripts\python.exe -m streamlit run interview_helper_local.py
pause
