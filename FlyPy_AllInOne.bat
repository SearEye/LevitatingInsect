:: ===================== FlyPy_AllInOne.bat (v3.17) =====================
@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ---- Resolve project directory (prefer CWD if it contains FlyAPI.py) ----
set "SCRIPT_DIR=%~dp0"
set "PROJ_DIR=%CD%"
if exist "%PROJ_DIR%\FlyAPI.py" (
  rem good
) else if exist "%SCRIPT_DIR%\FlyAPI.py" (
  set "PROJ_DIR=%SCRIPT_DIR%"
) else (
  echo [ERROR] Could not locate FlyAPI.py in "%CD%" or "%SCRIPT_DIR%".
  echo         Please run this from the project folder.
  goto :end_fail
)

pushd "%PROJ_DIR%"
if not exist "logs" mkdir "logs"
set "LOG=logs\FlyPy_AIO.log"

:: ---- Parse simple args ----
set "DO_SETUP=1"
set "DO_RUN=1"
set "NO_PAUSE="
for %%A in (%*) do (
  if /I "%%~A"=="/setuponly" set "DO_RUN=0"
  if /I "%%~A"=="/runonly"   set "DO_SETUP=0"
  if /I "%%~A"=="/nopause"   set "NO_PAUSE=1"
)

echo ===============================================
echo  FlyPy All-In-One (v3.17)
echo  Project: %PROJ_DIR%
echo  Log: %PROJ_DIR%\%LOG%
echo ===============================================

:: ---- Find/create venv ----
set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_PIP=.venv\Scripts\pip.exe"
if "%DO_SETUP%"=="1" (
  if not exist ".venv\Scripts\activate.bat" (
    echo [SETUP] Creating venv...>>"%LOG%"
    where python >nul 2>&1
    if errorlevel 1 (
      echo [ERROR] No system Python found. Install Python 3.10+ and retry.>>"%LOG%"
      echo [ERROR] No system Python found. Install Python 3.10+ and retry.
      goto :end_fail
    )
    for /f "delims=" %%P in ('where python') do (
      set "SYS_PY=%%~fP"
      goto :have_py
    )
    :have_py
    "%SYS_PY%" -m venv ".venv" >>"%LOG%" 2>&1 || (
      echo [ERROR] venv creation failed.>>"%LOG%"
      echo [ERROR] venv creation failed.
      goto :end_fail
    )
  )

  echo [SETUP] Upgrading pip tooling...>>"%LOG%"
  "%VENV_PY%" -m pip install --upgrade pip setuptools wheel >>"%LOG%" 2>&1

  :: Choose requirements file (lock preferred)
  set "REQ=requirements.lock.txt"
  if not exist "%REQ%" set "REQ=requirements.txt"
  if not exist "%REQ%" (
    echo [WARN ] No requirements file found. Skipping Python deps.>>"%LOG%"
  ) else (
    echo [SETUP] Installing %REQ% ...>>"%LOG%"
    "%VENV_PIP%" install -r "%REQ%" --upgrade >>"%LOG%" 2>&1 || (
      echo [WARN ] pip install returned an error; continuing (GUI may still run).>>"%LOG%"
    )
  )

  :: Optional: local PySpin wheel
  if defined FLYPY_SPIN_WHEEL (
    if exist %FLYPY_SPIN_WHEEL% (
      echo [SETUP] Installing PySpin wheel: %FLYPY_SPIN_WHEEL%>>"%LOG%"
      "%VENV_PIP%" install %FLYPY_SPIN_WHEEL% >>"%LOG%" 2>&1
    ) else (
      echo [WARN ] FLYPY_SPIN_WHEEL path not found: %FLYPY_SPIN_WHEEL%>>"%LOG%"
    )
  )

  :: Try to ensure PsychoPy (best-effort, will not fail the run)
  "%VENV_PY%" -c "import psychopy" >nul 2>&1 || (
    echo [SETUP] Installing PsychoPy (best-effort)...>>"%LOG%"
    "%VENV_PIP%" install --prefer-binary "psychopy==2025.1.1" >>"%LOG%" 2>&1
  )
)

:: ---- Prime Spinnaker PATH if SDK present ----
set "SPINROOT="
if defined SPINNAKER_PATH set "SPINROOT=%SPINNAKER_PATH%"
if not defined SPINROOT if exist "C:\Program Files\Teledyne FLIR\Spinnaker\bin64" set "SPINROOT=C:\Program Files\Teledyne FLIR\Spinnaker"
if not defined SPINROOT if exist "C:\Program Files\FLIR Systems\Spinnaker\bin64"  set "SPINROOT=C:\Program Files\FLIR Systems\Spinnaker"
if defined SPINROOT (
  set "PATH=%SPINROOT%\bin64;%SPINROOT%\lib64;%PATH%"
  echo [PATH ] Using Spinnaker runtime at: %SPINROOT%>>"%LOG%"
)

:: ---- Run FlyAPI ----
if "%DO_RUN%"=="1" (
  if not exist "%VENV_PY%" (
    echo [ERROR] venv missing. Run with /setuponly first.>>"%LOG%"
    echo [ERROR] venv missing. Run with /setuponly first.
    goto :end_fail
  )
  echo [RUN ] Launching FlyAPI.py ...>>"%LOG%"
  "%VENV_PY%" "FlyAPI.py" %* 2>>"%LOG%"
  set "RC=%ERRORLEVEL%"
  echo [EXIT] FlyAPI.py returned %RC%>>"%LOG%"
  echo.
  echo [Done] Exit code %RC%. See %LOG% for details.
)

goto :end_ok

:end_fail
echo *** ERROR: See %LOG% for details. ***
if not defined NO_PAUSE pause
popd
exit /b 1

:end_ok
if not defined NO_PAUSE pause
popd
exit /b 0
:: =================== end FlyPy_AllInOne.bat (v3.17) ===================
