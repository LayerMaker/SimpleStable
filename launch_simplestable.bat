@echo off
setlocal

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python first.
    pause
    exit /b 1
)

:: Remove existing virtual environment
if exist "venv" (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

:: Create new virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install requirements
echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Launch the application
echo Launching SimpleStable...
python app.py

:: Keep the window open if there's an error
if errorlevel 1 (
    echo An error occurred while running the application.
    pause
)

deactivate
