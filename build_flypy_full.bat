@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =====================================================================
REM FlyPy-Full one-folder builder (Windows)
REM - Creates venv
REM - Installs requirements + PsychoPy + PyInstaller
REM - Downloads CH340/CH341 driver to temp\drivers\CH341SER.EXE
REM - Builds one-folder app with PyInstaller using FlyPy_full.spec
REM - Copies README + LICENSE and bundles temp\ into dist\FlyPy\
REM =====================================================================

set PY=py -3.10
if "%1"=="/py38" set PY=py -3.8

echo.
echo [1/6] Creating/activating virtual environment...
%PY% -m venv .venv || goto :die
call .venv\Scripts\activate || goto :die

echo.
echo [2/6] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel || goto :die

echo.
echo [3/6] Installing core requirements...
if exist requirements.txt (
  pip install -r requirements.txt || goto :die
) else (
  echo (no requirements.txt found, continuing)
)

echo.
echo [4/6] Installing PsychoPy (full build) and PyInstaller...
REM You can pin a version if you prefer: pip install "psychopy==2025.2.1"
pip install psychopy pyinstaller || goto :die

echo.
echo [5/6] Preparing temp assets (drivers, helper scripts)...
set TEMP_DIR=%CD%\temp
set DRV_DIR=%TEMP_DIR%\drivers
set SCR_DIR=%TEMP_DIR%\scripts
mkdir "%DRV_DIR%" 2>nul
mkdir "%SCR_DIR%" 2>nul

REM --- Download WCH CH340/CH341 USB-Serial Windows driver (official) ---
set CH34X_EXE=%DRV_DIR%\CH341SER.EXE
if not exist "%CH34X_EXE%" (
  echo   Downloading CH340/CH341 driver (WCH, EN site)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference='SilentlyContinue';" ^
    "try { Invoke-WebRequest 'https://www.wch-ic.com/downloads/CH341SER_EXE.html' -OutFile '%CH34X_EXE%' -UseBasicParsing } catch { $false }" 1>nul
)

if not exist "%CH34X_EXE%" (
  echo   EN mirror failed, trying CN mirror...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference='SilentlyContinue';" ^
    "try { Invoke-WebRequest 'https://www.wch.cn/downloads/CH341SER_EXE.html' -OutFile '%CH34X_EXE%' -UseBasicParsing } catch { $false }" 1>nul
)

if not exist "%CH34X_EXE%" (
  echo   WARNING: Could not download CH341SER.EXE automatically.
  echo   -> Manually download from the official page and place as:
  echo      %CH34X_EXE%
  echo   Official pages: 
  echo      https://www.wch-ic.com/downloads/CH341SER_EXE.html
  echo      https://www.wch.cn/downloads/CH341SER_EXE.html
) else (
  echo   Driver saved to "%CH34X_EXE%"
)

REM --- Write temp\README and installer helper (these ship inside the app) ---
> "%TEMP_DIR%\README-FIRST-RUN.txt" (
  echo FlyPy — Temp Assets
  echo ====================
  echo This folder ships with the portable app for convenience.
  echo
  echo 1) If your Elegoo/CH340 board is not recognized on Windows,
  echo    run: temp\scripts\install_drivers.bat
  echo    (This launches the official WCH CH340/CH341 USB-serial driver installer.)
  echo
  echo 2) You can delete this folder after installing drivers; it is not
  echo    required for FlyPy to run.
)

> "%SCR_DIR%\install_drivers.bat" (
  echo @echo off
  echo setlocal
  echo set DRIVER=%%~dp0..\drivers\CH341SER.EXE
  echo if not exist "%%DRIVER%%" (
  echo   echo CH340/CH341 driver not found at %%DRIVER%% 
  echo   echo Please download from the official WCH page and place it there.
  echo   pause
  echo   exit /b 1
  echo )
  echo echo Launching CH340/CH341 driver installer...
  echo start "" "%%DRIVER%%"
)

echo.
echo [6/6] Building with PyInstaller (one-folder)...
REM Use the .spec for reproducibility (collects PsychoPy/cv2/etc + bundles temp/)
pyinstaller --noconfirm --clean FlyPy_full.spec || goto :die

echo.
echo Build complete.
echo -----------------------------------------
echo Portable app:   dist\FlyPy\FlyPy.exe
echo Bundled assets: dist\FlyPy\temp\ (drivers, scripts, README)
echo -----------------------------------------
exit /b 0

:die
echo.
echo Build failed. See messages above.
exit /b 1
