@echo off
setlocal ENABLEEXTENSIONS

REM Run from this script's folder
cd /d "%~dp0"

REM Ensure the main batch exists
if not exist "FlyPy.bat" (
  echo [ERROR] FlyPy.bat not found in "%~dp0"
  echo Make sure this Start file is in the same folder as FlyPy.bat.
  pause
  exit /b 1
)

REM Default launch args if none provided to Start
if "%~1"=="" (
  set "LAUNCH_ARGS=--simulate"
) else (
  set "LAUNCH_ARGS=%*"
)

echo [START] Running setup...
call FlyPy.bat setup
if errorlevel 1 (
  echo [START] Setup failed. See messages above.
  pause
  exit /b 1
)

echo [START] Launching FlyPy (args: %LAUNCH_ARGS%)...
call FlyPy.bat launch %LAUNCH_ARGS%
set "ERR=%ERRORLEVEL%"

if not "%ERR%"=="0" (
  echo [START] Launch failed with exit code %ERR%.
) else (
  echo [START] FlyPy exited successfully.
)

echo.
echo Press any key to close...
pause >nul
endlocal & exit /b %ERR%
