@echo off
title AI Smoking Area Tracking System
echo ========================================
echo AI Smoking Area Tracking System Starting...
echo ========================================
echo.

:: Python check
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python and add to PATH.
    pause
    exit /b 1
)

:: Install required packages
echo Checking required packages...
echo Installing packages (InsightFace for face recognition)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages!
    echo Check requirements.txt file.
    pause
    exit /b 1
)

:: Initialize database
echo Initializing database...
python -c "import db_manager; db_manager.init_db()" >nul 2>&1

:: Start Flask application
echo Starting Flask application...
echo Open http://localhost:5000 in your browser.
echo Close this window to stop.
echo ========================================
echo.

python app.py

pause
