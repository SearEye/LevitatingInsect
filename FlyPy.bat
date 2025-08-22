@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =====================================================================
REM FlyPy.bat — One script to SETUP, LAUNCH, BUILD (Full w/ PsychoPy),
REM              DOWNLOAD/RUN DRIVERS, FREEZE REQUIREMENTS, and CLEAN.
REM
REM USAGE:
REM   FlyPy.bat help
REM   FlyPy.bat setup           (create venv + pip install -r requirements.txt)
REM   FlyPy.bat setup-full      (setup + install PsychoPy)
REM   FlyPy.bat launch          (run FlyAPI.py in venv)
REM   FlyPy.bat build-full      (PyInstaller one-folder build WITH PsychoPy; no .spec needed)
REM   FlyPy.bat drivers         (download CH340/CH341 driver to temp\drivers and launch it)
REM   FlyPy.bat lock            (pip freeze -> requirements.lock.txt)
REM   FlyPy.bat clean           (remove build/, dist/)
REM   FlyPy.bat clean-all       (also remove .venv/)
REM
REM Optional: set FLYPY_PYTHON to override launcher (e.g. "py -3.10" or full path)
REM Default Python launcher is "py -3.10" (good for PsychoPy).
REM =====================================================================

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

if not defined FLYPY_PYTHON (
  set "PY=py -3.10"
) else (
  set "PY=%FLYPY_PYTHON%"
)

set "VENV=.venv"
set "PYTHON_EXE=%ROOT%%VENV%\Scripts\python.exe"
set "PIP_EXE=%ROOT%%VENV%\Scripts\pip.exe"

REM ---------- dispatch ----------
if /I "%~1"=="help"        goto :help
if /I "%~1"=="setup"       goto :setup
if /I "%~1"=="setup-full"  goto :setup_full
if /I "%~1"=="launch"      goto :launch
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
echo   setup-full    setup + install PsychoPy
echo   launch        Run FlyAPI.py (GUI) in the venv
echo   build-full    Build portable one-folder app with PsychoPy (PyInstaller)
echo   drivers       Download CH340/CH341 driver to temp\drivers and launch it
echo   lock          Freeze exact versions to requirements.lock.txt
echo   clean         Remove build/ and dist/
echo   clean-all     Remove build/, dist/, and .venv/
echo.
echo Examples:
echo   FlyPy.bat setup
echo   FlyPy.bat launch
echo   FlyPy.bat build-full
echo   FlyPy.bat drivers
echo =====================================================================
goto :eof


:setup
echo.
echo [SETUP] Creating virtual environment...
%PY% -m venv "%VENV%" || goto :die

call "%VENV%\Scripts\activate" || goto :die

echo.
echo [SETUP] Upgrading pip/setuptools/wheel...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel || goto :die

if exist "requirements.txt" (
  echo.
  echo [SETUP] Installing requirements.txt ...
  "%PIP_EXE%" install -r requirements.txt || goto :die
) else (
  echo [SETUP] No requirements.txt found. Installing core deps directly...
  "%PIP_EXE%" install numpy opencv-python PyQt5 pyserial || goto :die
)
echo.
echo [SETUP] Done.
goto :eof


:setup_full
call :setup || goto :die
echo.
echo [SETUP-FULL] Installing PsychoPy (full build)...
"%PIP_EXE%" install psychopy || goto :die
echo [SETUP-FULL] Done.
goto :eof


:launch
if not exist "%PYTHON_EXE%" (
  echo [LAUNCH] No venv found. Running setup first...
  call :setup || goto :die
)
echo.
echo [LAUNCH] Starting FlyAPI.py ...
call "%VENV%\Scripts\activate"
"%PYTHON_EXE%" "FlyAPI.py"
goto :eof


:build_full
REM Ensure venv + deps + PyInstaller + PsychoPy are installed
if not exist "%PYTHON_EXE%" (
  echo [BUILD] No venv found. Running setup-full first...
  call :setup_full || goto :die
) else (
  call "%VENV%\Scripts\activate"
  "%PIP_EXE%" install --upgrade pip setuptools wheel || goto :die
  "%PIP_EXE%" install psychopy pyinstaller || goto :die
)

REM Prepare temp assets (drivers + helper scripts) that will be bundled
call :prepare_temp_assets

echo.
echo [BUILD] Running PyInstaller (one-folder, spec-less)...
pyinstaller ^
  --noconfirm --clean --windowed ^
  --name FlyPy ^
  --hidden-import serial.tools.list_ports ^
  --collect-all PyQt5 --collect-all cv2 --collect-all psychopy ^
  --collect-all pyglet --collect-all sounddevice --collect-all soundfile ^
  --add-data "README.md;." --add-data "LICENSE;." ^
  --add-data "temp;temp" ^
  FlyAPI.py || goto :die

REM Extra safety: if add-data missed temp\, copy it
if exist "dist\FlyPy" (
  xcopy /E /I /Y "temp" "dist\FlyPy\temp" >nul 2>nul
)

echo.
echo [BUILD] Build complete.
echo -----------------------------------------
echo Portable app:   dist\FlyPy\FlyPy.exe
echo Bundled assets: dist\FlyPy\temp\  (drivers, scripts, README)
echo -----------------------------------------
goto :eof


:drivers
REM Download and launch CH340/CH341 driver
call :fetch_driver
set "DRV=%ROOT%temp\drivers\CH341SER.EXE"
if exist "%DRV%" (
  echo [DRIVERS] Launching CH340/CH341 driver installer...
  start "" "%DRV%"
) else (
  echo [DRIVERS] Driver EXE not found. See the messages above for manual URLs.
)
goto :eof


:lock
if not exist "%PYTHON_EXE%" (
  echo [LOCK] No venv found. Running setup first...
  call :setup || goto :die
)
call "%VENV%\Scripts\activate"
echo [LOCK] Freezing versions to requirements.lock.txt ...
"%PIP_EXE%" freeze > "requirements.lock.txt" || goto :die
echo [LOCK] Wrote requirements.lock.txt
goto :eof


:clean
echo [CLEAN] Removing build/ and dist/ ...
rmdir /S /Q "build" 2>nul
rmdir /S /Q "dist" 2>nul
echo [CLEAN] Done.
goto :eof


:clean_all
call :clean
echo [CLEAN-ALL] Removing .venv/ ...
rmdir /S /Q "%VENV%" 2>nul
echo [CLEAN-ALL] Done.
goto :eof


REM -------------------------------
REM helpers
REM -------------------------------
:prepare_temp_assets
set "TEMP_DIR=%ROOT%temp"
set "DRV_DIR=%TEMP_DIR%\drivers"
set "SCR_DIR=%TEMP_DIR%\scripts"
mkdir "%DRV_DIR%" 2>nul
mkdir "%SCR_DIR%" 2>nul

REM Write README-FIRST-RUN (ships inside portable app)
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

REM Generate an on-demand driver installer script inside temp\scripts\
> "%SCR_DIR%\install_drivers.bat" (
  echo @echo off
  echo setlocal
  echo set DRIVER=%%~dp0..\drivers\CH341SER.EXE
  echo if not exist "%%DRIVER%%" (
  echo   echo CH340/CH341 driver not found at %%DRIVER%%
  echo   echo Please download from the official WCH page and place it there:
  echo   echo   https://www.wch-ic.com/downloads/CH341SER_EXE.html
  echo   echo   https://www.wch.cn/downloads/CH341SER_EXE.html
  echo   pause
  echo   exit /b 1
  echo )
  echo echo Launching CH340/CH341 driver installer...
  echo start "" "%%DRIVER%%"
)

REM Always try to fetch the driver EXE so it's present in a fresh build
call :fetch_driver
goto :eof


:fetch_driver
set "CH34X_EXE=%ROOT%temp\drivers\CH341SER.EXE"
if exist "%CH34X_EXE%" (
  echo [DRIVERS] Driver already present at "%CH34X_EXE%"
  goto :eof
)
echo [DRIVERS] Downloading CH340/CH341 driver (WCH, EN site)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ProgressPreference='SilentlyContinue';" ^
  "try { Invoke-WebRequest 'https://www.wch-ic.com/downloads/CH341SER_EXE.html' -OutFile '%CH34X_EXE%' -UseBasicParsing } catch { $false }" 1>nul

if not exist "%CH34X_EXE%" (
  echo [DRIVERS] EN mirror failed, trying CN mirror...
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference='SilentlyContinue';" ^
    "try { Invoke-WebRequest 'https://www.wch.cn/downloads/CH341SER_EXE.html' -OutFile '%CH34X_EXE%' -UseBasicParsing } catch { $false }" 1>nul
)

if not exist "%CH34X_EXE%" (
  echo [DRIVERS] Could not download CH341SER.EXE automatically.
  echo          Please download from either official page and save as:
  echo            %CH34X_EXE%
  echo          EN: https://www.wch-ic.com/downloads/CH341SER_EXE.html
  echo          CN: https://www.wch.cn/downloads/CH341SER_EXE.html
) else (
  echo [DRIVERS] Driver saved to "%CH34X_EXE%"
)
goto :eof


:die
echo.
echo *** ERROR: A step failed. See messages above. ***
exit /b 1
