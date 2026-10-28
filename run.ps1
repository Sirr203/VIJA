# Run script for Vija on Windows PowerShell
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "        VIJA - VIETNAMESE <-> JAPANESE TRANSLATION      " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".venv")) {
    Write-Host "[!] Virtual environment (.venv) not found. Creating virtual environment..." -ForegroundColor Yellow
    py -3.12 -m venv .venv
    Write-Host "[i] Installing dependencies..." -ForegroundColor Yellow
    & .venv\Scripts\python -m pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
    Write-Host "[!] .env file not found. Generating from .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "[!] Please open .env and set your GROQ_API_KEY (from console.groq.com) before running!" -ForegroundColor Red
    return
}

Write-Host "[i] Starting server at http://localhost:8000 ..." -ForegroundColor Green
Write-Host "[i] Press Ctrl+C to stop the server." -ForegroundColor Gray
Write-Host ""

& .venv\Scripts\python -m uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000
