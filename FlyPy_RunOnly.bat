@echo off
setlocal enableextensions

REM === FlyPy Safe Launcher with Backups ===
set D=%~dp0
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do (set dstr=%%d%%b%%c)
for /f "tokens=1-2 delims=: " %%a in ("%time%") do (set tstr=%%a%%b)
set tstr=%tstr::=%
set tstr=%tstr: =0%
set BK=%D%Backups\%dstr%_%tstr%

if not exist "%D%.venv\Scripts\python.exe" (
    echo [ERR] Python venv missing at %D%.venv\Scripts\python.exe
    pause
    exit /b 1
)

mkdir "%BK%" >nul 2>nul
copy "%D%FlyAPI.py" "%BK%\FlyAPI.py" >nul

REM Qt backend (ANGLE recommended)
if "%FLYPY_QT_OPENGL%"=="" set FLYPY_QT_OPENGL=angle

call "%D%.venv\Scripts\activate"
python "%D%FlyAPI.py"
set rc=%ERRORLEVEL%
echo [EXIT] FlyAPI.py returned %rc%
pause
exit /b %rc%
