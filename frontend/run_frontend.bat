@echo off
title Cert8fy Biometric Workstation - Frontend Server (Port 3000)
cd /d "%~dp0"
echo ==========================================================
echo Starting Cert8fy Biometric UI on http://localhost:3000...
echo ==========================================================
npm run dev
pause
