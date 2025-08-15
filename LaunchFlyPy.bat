@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ======================================================
REM LaunchFlyPy.bat — Sync + Activate venv + Install + Run
REM - Fast-forward git pull on current branch (if git exists)
REM - Update submodules (if any)
REM - Ensure .venv, activate, upgrade pip
REM - If requirements.txt missing: auto-call Install.bat
REM - Install from requirements.txt and write requirements.lock.txt
REM - Run entrypoint (defaults to FlyAPI.py; override via FLYPY_ENTRY or first arg)
REM ======================================================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

REM Choose Python launcher
where py >nul 2>&1
if %ERRORLEVEL%==0 ( set "PYTHON=py" ) else ( set "PYTHON=python" )

REM Resolve entrypoint (default to FlyAPI.py in this repo)
set "ENTRYPOINT=FlyAPI.py"
if not "%FLYPY_ENTRY%"=="" set "ENTRYPOINT=%FLYPY_ENTRY%"
if not "%~1"=="" set "ENTRYPOINT=%~1"

REM ----- Git sync (optional) -----
where git >nul 2>&1
if %ERRORLEVEL%==0 (
  git rev-parse --is-inside-work-tree >nul 2>&1
  if %ERRORLEVEL%==0 (
    echo [INFO] Fetching latest changes...
    git fetch --all --prune
    for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD') do set "CURBRANCH=%%i"
    if "%CURBRANCH%"=="" set "CURBRANCH=main"
    echo [INFO] Pulling fast-forward on "%CURBRANCH%" ...
    git pull --ff-only origin "%CURBRANCH%"
    if %ERRORLEVEL% NEQ 0 (
      echo [WARN] Fast-forward pull failed (local changes?). Continuing without sync...
    )
    if exist ".gitmodules" (
      echo [INFO] Updating submodules...
      git submodule update --init --recursive
    )
  ) else (
    echo [WARN] Not a Git repository. Skipping sync.
  )
) else (
  echo [WARN] Git not found on PATH. Skipping sync.
)

REM ----- Virtual environment -----
if not exist .venv (
  echo [INFO] Creating virtual environment in .venv ...
  %PYTHON% -m venv .venv
  if ERRORLEVEL 1 (
    echo [ERROR] Failed to create virtual environment.
    popd & endlocal & exit /b 1
  )
)

call .venv\Scripts\activate
if ERRORLEVEL 1 (
  echo [ERROR] Failed to activate virtual environment.
  popd & endlocal & exit /b 1
)

echo [INFO] Upgrading pip, setuptools, wheel ...
python -m pip install --upgrade pip setuptools wheel

REM ----- Auto-install if requirements.txt missing -----
if not exist requirements.txt (
  echo [INFO] requirements.txt not found — invoking Install.bat to generate/install...
  if exist "%SCRIPT_DIR%Install.bat" (
    call "%SCRIPT_DIR%Install.bat"
    if ERRORLEVEL 1 (
      echo [ERROR] Install.bat failed; cannot continue.
      popd & endlocal & exit /b 1
    )
  ) else (
    echo [ERROR] Install.bat not found at: "%SCRIPT_DIR%"
    popd & endlocal & exit /b 1
  )
)

REM ----- Install from requirements.txt -----
echo [INFO] Installing from requirements.txt ...
pip install -r requirements.txt
if ERRORLEVEL 1 (
  echo [ERROR] pip install failed. See output above.
  popd & endlocal & exit /b 1
)

echo [INFO] Writing exact versions to requirements.lock.txt ...
pip freeze > requirements.lock.txt

REM Verify entrypoint exists
if not exist "%ENTRYPOINT%" (
  echo [ERROR] Entrypoint not found: "%ENTRYPOINT%"
  echo         Override with:  set FLYPY_ENTRY=FlyAPI.py ^& LaunchFlyPy.bat
  echo         Or:              LaunchFlyPy.bat FlyAPI.py
  popd & endlocal & exit /b 1
)

REM Run the app; pass through remaining args (shift if first arg was entrypoint)
if not "%~1"=="" shift
echo [INFO] Running: python "%ENTRYPOINT%" %*
python "%ENTRYPOINT%" %*
set APP_RC=%ERRORLEVEL%

popd
endlocal & exit /b %APP_RC%
