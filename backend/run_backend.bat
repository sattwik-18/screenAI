@echo off
title Cert8fy Biometric Engine - Backend Server (Port 8000)
cd /d "%~dp0"
set PYTHONPATH=%cd%

echo ==========================================================
echo Starting Cert8fy Biometric AI Backend on port 8000...
echo Directory: %PYTHONPATH%
echo ==========================================================

:: Auto-detect correct Python environment with required packages
set "PYTHON_CMD="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :run_server
)
if exist "C:\Users\SATTWIK\AppData\Local\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD="C:\Users\SATTWIK\AppData\Local\Programs\Python\Python312\python.exe"
    goto :run_server
)

:: Fallback to Windows Python Launcher
py -3.12 -c "import uvicorn" >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.12
    goto :run_server
)

:: Fallback to system PATH python
set PYTHON_CMD=python

:run_server
echo Using Python runtime: %PYTHON_CMD%
echo.
%PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause

