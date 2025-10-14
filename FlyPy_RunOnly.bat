
@echo off
setlocal EnableExtensions EnableDelayedExpansion
title FlyPy Run Only
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "LOG_DIR=%ROOT%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1
set "LOG_FILE=%LOG_DIR%\FlyPy_Run.log"
break > "%LOG_FILE%"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%VENV_PY%" ( echo venv missing. Run FlyPy_SetupOnly.bat first. & exit /b 1 )
if exist "%ROOT%FlyAPI.py" (
  "%VENV_PY%" "%ROOT%FlyAPI.py" %* >> "%LOG_FILE%" 2>&1
) else if exist "%ROOT%FlyAPI.exe" (
  "%ROOT%FlyAPI.exe" %* >> "%LOG_FILE%" 2>&1
) else (
  echo FlyAPI.py / FlyAPI.exe not found. >> "%LOG_FILE%"
  exit /b 1
)
echo Run complete. Log: "%LOG_FILE%"
endlocal & exit /b 0
