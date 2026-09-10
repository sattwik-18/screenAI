@echo off
title Cert8fy - Fullstack Biometric Verification Platform
cd /d "%~dp0"
echo ==========================================================
echo   Cert8fy Advanced Facial Identity Verification Platform
echo ==========================================================
echo.
echo Launching Backend and Frontend in separate windows...
echo.

start "Cert8fy Backend" "%~dp0backend\run_backend.bat"
timeout /t 2 /nobreak >nul
start "Cert8fy Frontend" "%~dp0frontend\run_frontend.bat"

echo ==========================================================
echo Services launched:
echo  - Backend API:  http://localhost:8000/api/v1/health
echo  - Frontend Web: http://localhost:3000/
echo ==========================================================
