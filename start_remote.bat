@echo off
REM Launches the local system-audio transcriber and a public Cloudflare tunnel.
REM Double-click this file, then read the URL from the "Tunnel" window.
cd /d "%~dp0"

REM ======================================================================
REM  SET YOUR PASSWORD HERE. Anyone opening the link must type this.
REM  Change it to whatever you want, then save this file.
set APP_PASSWORD=changeme123
REM ======================================================================

echo Starting the transcriber app...
start "Transcriber" .venv\Scripts\python.exe -m streamlit run interview_helper_local.py ^
  --server.headless true --server.port 8501 ^
  --server.enableCORS false --server.enableXsrfProtection false

echo Waiting for the app to come up...
timeout /t 7 >nul

echo Starting the public tunnel...
start "Tunnel" cloudflared.exe tunnel --url http://localhost:8501 --no-autoupdate

echo.
echo ============================================================
echo  Your app is starting.
echo  Look at the "Tunnel" window that just opened - it prints a
echo  line like:  https://something-random.trycloudflare.com
echo  Open THAT link on any computer or phone.
echo.
echo  Keep BOTH windows open while you use it.
echo  Closing them (or shutting down this PC) stops remote access.
echo ============================================================
echo.
pause
