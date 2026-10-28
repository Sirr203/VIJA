@echo off
chcp 65001 >nul
echo ========================================================
echo             VIJA - VIETNAMESE ^<-^> JAPANESE
echo ========================================================
echo.

if not exist .venv (
    echo [!] Virtual environment (.venv) not found. Creating virtual environment...
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
    echo [i] Installing required dependencies...
    .venv\Scripts\python -m pip install -r requirements.txt
)

if not exist .env (
    echo [!] .env file not found. Generating from .env.example...
    copy .env.example .env
    echo [!] Please open .env and enter your GROQ_API_KEY (from console.groq.com) before running!
    pause
    exit /b 1
)

echo [i] Starting server at http://localhost:8000 ...
echo [i] Press Ctrl+C to stop the server.
echo.
.venv\Scripts\python -m uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000
pause
