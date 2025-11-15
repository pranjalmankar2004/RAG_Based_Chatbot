@echo off
REM Legal RAG Chatbot - 24/7 Service
REM Auto-restarts on crash

:loop
cd /d "%~dp0"
echo.
echo ===============================================
echo Legal RAG Chatbot - 24/7 Service
echo ===============================================
echo Starting backend on port 8001...
echo Time: %date% %time%
echo.

python backend/main.py

echo.
echo ===============================================
echo ERROR: Server stopped
echo Restarting in 5 seconds...
echo ===============================================
timeout /t 5 /nobreak

goto loop
