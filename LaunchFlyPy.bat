@echo off
setlocal

REM ======================================================
REM LaunchFlyPy.bat — Safe launcher (no for/quoted subshells)
REM Usage:
REM   LaunchFlyPy.bat           -> normal run (defaults to FlyAPI.py)
REM   LaunchFlyPy.bat /debug    -> verbose + pause on errors
REM   LaunchFlyPy.bat Main.py   -> run a different entrypoint
REM   set FLYPY_ENTRY=Main.py & LaunchFlyPy.bat
REM ======================================================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

REM --- Parse /debug flag (first arg) ---
set "DEBUG=0"
if /I "%~1"=="/debug" set "DEBUG=1" & shift
if /I "%~1"=="--debug" set "DEBUG=1" & shift

REM --- Pick Python launcher ---
where py >nul 2>&1 && (set "PY=py") || (set "PY=python")

REM --- Resolve entrypoint (default to FlyAPI.py) ---
set "ENTRYPOINT=FlyAPI.py"
if not "%FLYPY_ENTRY%"=="" set "ENTRYPOINT=%FLYPY_ENTRY%"
if not "%~1"=="" set "ENTRYPOINT=%~1"

REM --- Optional Git sync (no branch parsing to avoid CMD quirks) ---
where git >nul 2>&1 && (
  git rev-parse --is-inside-work-tree >nul 2>&1 && (
    echo [INFO] Syncing repository...
    git fetch --all --prune
    git pull --ff-only
    if exist ".gitmodules" git submodule update --init --recursive
  )
)

REM --- Ensure virtual environment ---
if not exist ".venv\Scripts\activate" (
  echo [INFO] Creating virtual environment in .venv ...
  %PY% -m venv .venv || (echo [ERROR] venv creation failed.& goto :fail)
)

call ".venv\Scripts\activate" || (echo [ERROR] venv activation failed.& goto :fail)

echo [INFO] Upgrading pip, setuptools, wheel ...
python -m pip install --upgrade pip setuptools wheel

REM --- Ensure requirements installed (auto-call Install.bat if missing) ---
if not exist requirements.txt (
  echo [INFO] requirements.txt not found — calling Install.bat ...
  if exist "Install.bat" (
    call "Install.bat" || goto :fail
  ) else (
    echo [ERROR] Install.bat is missing and no requirements.txt available.
    goto :fail
  )
)

echo [INFO] Installing from requirements.txt ...
pip install -r requirements.txt || goto :fail

echo [INFO] Writing exact versions to requirements.lock.txt ...
pip freeze > requirements.lock.txt

REM --- Verify entrypoint ---
if not exist "%ENTRYPOINT%" (
  echo [ERROR] Entrypoint not found: "%ENTRYPOINT%"
  echo         Override with:  set FLYPY_ENTRY=MyMain.py ^& LaunchFlyPy.bat
  echo         Or:              LaunchFlyPy.bat MyMain.py
  goto :fail
)

REM --- Debug verbosity (optional) ---
if "%DEBUG%"=="1" (
  set QT_DEBUG_PLUGINS=1
  set PYTHONFAULTHANDLER=1
  echo [INFO] Debug mode ON (Qt plugin logs + faulthandler).
)

echo [INFO] Running: python "%ENTRYPOINT%"
python "%ENTRYPOINT%"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" goto :fail
echo [OK] Application exited normally.
if "%DEBUG%"=="1" pause
popd & endlocal & exit /b 0

:fail
echo.
echo [FAIL] Exit code: %ERRORLEVEL%
echo If the window closed instantly, run again as:  LaunchFlyPy.bat /debug
if "%DEBUG%"=="1" pause
popd & endlocal & exit /b 1
