@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =====================================================================
REM FlyPy.bat — Setup, Launch, Build, Drivers, Freeze, Clean
REM
REM USAGE:
REM   FlyPy.bat help
REM   FlyPy.bat setup              (create venv + pip install -r requirements.txt)
REM   FlyPy.bat setup-full         (setup + install PsychoPy)
REM   FlyPy.bat launch [--simulate] [--prewarm-stim]  (run FlyAPI.py; extra args are forwarded)
REM   FlyPy.bat build-full         (PyInstaller one-folder build WITH PsychoPy)
REM   FlyPy.bat drivers            (download CH340/CH341 driver to temp\drivers and launch it)
REM   FlyPy.bat lock               (pip freeze -> requirements.lock.txt)
REM   FlyPy.bat clean              (remove build/, dist/)
REM   FlyPy.bat clean-all          (remove venv/, build/, dist/)
REM =====================================================================

set ROOT=%~dp0
pushd "%ROOT%"

set VENV_DIR=%ROOT%venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set PIP_EXE=%VENV_DIR%\Scripts\pip.exe

if /I "%~1"=="help"        goto :help
if /I "%~1"=="setup"       goto :setup
if /I "%~1"=="setup-full"  goto :setup_full
if /I "%~1"=="launch"      shift & goto :launch
if /I "%~1"=="build-full"  goto :build_full
if /I "%~1"=="drivers"     goto :drivers
if /I "%~1"=="lock"        goto :lock
if /I "%~1"=="clean"       goto :clean
if /I "%~1"=="clean-all"   goto :clean_all

echo.
echo FlyPy.bat — no command given. Showing help...
echo.
goto :help

:help
echo =====================================================================
echo FlyPy — All-in-one script
echo ---------------------------------------------------------------------
echo   setup         Create venv and install requirements.txt
echo   setup-full    Create venv and install requirements incl. PsychoPy
echo   launch        Run FlyAPI.py (extra args forwarded to Python)
echo                 Examples:
echo                   FlyPy.bat launch --simulate
echo                   FlyPy.bat launch --simulate --prewarm-stim
echo   build-full    PyInstaller one-folder build WITH PsychoPy
echo   drivers       Download CH340/CH341 driver into temp\drivers and run
echo   lock          pip freeze ^> requirements.lock.txt
echo   clean         Remove build/, dist/
echo   clean-all     Remove venv/, build/, dist/
echo =====================================================================
goto :eof

:setup
if exist "%VENV_DIR%" (
  echo [SETUP] venv exists.
) else (
  echo [SETUP] Creating venv...
  py -3 -m venv "%VENV_DIR%" || goto :die
)
echo [SETUP] Installing core requirements...
"%PIP_EXE%" install --upgrade pip || goto :die
"%PIP_EXE%" install -r requirements.txt || goto :die
echo [SETUP] Done.
goto :eof

:setup_full
call :setup || goto :die
echo [SETUP-FULL] Installing PsychoPy (for Python ^< 3.11)...
"%PIP_EXE%" install "psychopy>=2023.2.3; python_version<'3.11'" || echo [SETUP-FULL] PsychoPy skipped or failed (ok on Python 3.11+).
goto :eof

:launch
if not exist "%PYTHON_EXE%" call :setup || goto :die
echo [LAUNCH] Starting FlyPy...
REM Forward all remaining arguments (%*) to Python so flags like --simulate work
"%PYTHON_EXE%" -u "%ROOT%FlyAPI.py" %* || goto :die
goto :eof

:build_full
if not exist "%PYTHON_EXE%" call :setup_full || goto :die
echo [BUILD] Building one-folder distribution WITH PsychoPy...
"%PIP_EXE%" install pyinstaller || goto :die
pyinstaller --noconfirm --clean ^
  --name FlyPy ^
  --onefile "%ROOT%FlyAPI.py" || goto :die
echo [BUILD] Done. See dist\.
goto :eof

:drivers
set DRV_DIR=%ROOT%temp\drivers
set CH34X_EXE=%DRV_DIR%\CH341SER.EXE
mkdir "%DRV_DIR%" 2>nul
echo [DRIVERS] Downloading CH340/CH341 driver...
powershell -NoLogo -NoProfile -Command ^
  "(New-Object Net.WebClient).DownloadFile('https://www.wch-ic.com/downloads/file/65.html','%CH34X_EXE%')" ^
  || echo [DRIVERS] Auto-download failed. Please download and save to: %CH34X_EXE%
if exist "%CH34X_EXE%" (
  echo [DRIVERS] Launching driver installer...
  start "" "%CH34X_EXE%"
) else (
  echo [DRIVERS] Could not find %CH34X_EXE%.
  echo          Please download from WCH and save as shown above.
)
goto :eof

:lock
if not exist "%PYTHON_EXE%" call :setup || goto :die
echo [LOCK] Freezing environment to requirements.lock.txt...
"%PIP_EXE%" freeze > "%ROOT%requirements.lock.txt" || goto :die
echo [LOCK] Done.
goto :eof

:clean
echo [CLEAN] Removing build/, dist/...
rmdir /s /q "%ROOT%build" 2>nul
rmdir /s /q "%ROOT%dist" 2>nul
echo [CLEAN] Done.
goto :eof

:clean_all
call :clean
echo [CLEAN-ALL] Removing venv/...
rmdir /s /q "%ROOT%venv" 2>nul
echo [CLEAN-ALL] Done.
goto :eof

:die
echo.
echo *** ERROR: A step failed. See messages above. ***
exit /b 1
