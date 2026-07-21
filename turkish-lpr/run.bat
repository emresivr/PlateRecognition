@echo off
setlocal

cd /d "%~dp0"

echo =========================================
echo Turkish LPR - First Run Setup
echo =========================================

REM 1. Update .env if it does not exist
if not exist .env (
    echo Creating .env from .env.example...
    copy .env.example .env
)

REM 2. Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM 3. Verify health
echo.
echo Running health check...
python -m app.main health

REM 4. Run detection test (will download models on first run)
echo.
echo Running detection test on test frame...
python -m app.main detect-frame --source data\test_frame.jpg

echo.
pause
