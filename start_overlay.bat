@echo off
REM Launches the see-through, always-on-top overlay version of the app.
REM Drag the window to move it. Press Alt+F4 to close.
cd /d "%~dp0"
.venv\Scripts\python.exe overlay.py
