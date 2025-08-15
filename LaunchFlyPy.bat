@echo off
setlocal EnableExtensions

REM ======================================================
REM LaunchFlyPy.bat — robust launcher with pause-on-fail + logging
REM Usage:
REM   LaunchFlyPy.bat            -> normal run (defaults to FlyAPI.py)
REM   LaunchFlyPy.bat /debug     -> extra verbosity
REM   LaunchFlyPy.bat Main.py    -> run a different entrypoint
REM   set FLYPY_ENTRY=Main.py & LaunchFlyPy.bat
REM Output: logs\launch.log contains the full transcript
REM ======================================================

set PYTHONUTF8=1
set PYTHONIOENCODING=UTF-8

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

if not exist "logs" mkdir "logs" >nul 2>&1
set "LOGFILE=logs\launch.log"
echo ======== %DATE% %TIME% ======== > "%LOGFILE%"

REM --- args (/debug or alternate entrypoint) ---
set "DEBUG=0"
if /I "%~1"=="/debug" set "DEBUG=1" & shift
if /I "%~1"=="--debug" set "DEBUG=1" & shift

REM --- pick Python launcher ---
where py >nul 2>&1
if %ERRORLEVEL%==0 ( set "PY=py" ) else ( set "PY=python" )

REM --- resolve entrypoint (default to FlyAPI.py) ---
set "ENTRYPOINT=FlyAPI.py"
if not "%FLYPY_ENTRY%"=="" set "ENTRYPOINT=%FLYPY_ENTRY%"
if not "%~1"=="" set "ENTRYPOINT=%~1"

call :log "[INFO] Starting LaunchFlyPy"
call :log "[INFO] Entrypoint: %ENTRYPOINT%"

REM --- optional Git sync (simple, no subshells) ---
where git >nul 2>&1
if %ERRORLEVEL%==0 (
  git rev-parse --is-inside-work-tree >nul 2>&1
  if %ERRORLEVEL%==0 (
    call :log "[INFO] Syncing repository..."
    git fetch --all --prune        >> "%LOGFILE%" 2>&1
    git pull --ff-only             >> "%LOGFILE%" 2>&1
    if exist ".gitmodules" git submodule update --init --recursive >> "%LOGFILE%" 2>&1
  ) else (
    call :log "[WARN] Not a Git repository. Skipping sync."
  )
) else (
  call :log "[WARN] Git not found on PATH. Skipping sync."
)

REM --- ensure virtual environment ---
if not exist ".venv\Scripts\activate" (
  call :log "[INFO] Creating virtual environment in .venv ..."
  %PY% -m venv .venv >> "%LOGFILE%" 2>&1 || goto :fail
)

call ".venv\Scripts\activate" >> "%LOGFILE%" 2>&1 || goto :fail

call :log "[INFO] Upgrading pip, setuptools, wheel ..."
python -m pip install --upgrade pip setuptools wheel >> "%LOGFILE%" 2>&1 || goto :fail

REM --- auto-install if requirements.txt missing ---
if not exist requirements.txt (
  call :log "[INFO] requirements.txt not found — calling Install.bat ..."
  if exist "Install.bat" (
    call "Install.bat" >> "%LOGFILE%" 2>&1 || goto :fail
  ) else (
    call :log "[ERROR] Install.bat is missing and no requirements.txt available."
    goto :fail
  )
)

REM --- install requirements ---
call :log "[INFO] Installing from requirements.txt ..."
pip install -r requirements.txt >> "%LOGFILE%" 2>&1 || goto :fail

call :log "[INFO] Writing exact versions to requirements.lock.txt ..."
pip freeze > requirements.lock.txt 2>> "%LOGFILE%"

REM --- verify entrypoint exists ---
if not exist "%ENTRYPOINT%" (
  call :log "[ERROR] Entrypoint not found: %ENTRYPOINT%"
  goto :fail
)

REM --- debug switches (optional) ---
if "%DEBUG%"=="1" (
  set QT_DEBUG_PLUGINS=1
  set PYTHONFAULTHANDLER=1
  call :log "[INFO] Debug mode ON (Qt plugin logs + faulthandler)."
)

call :log "[INFO] Running: python %ENTRYPOINT%"
python -X faulthandler -u "%ENTRYPOINT%" >> "%LOGFILE%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :fail

call :log "[OK] Application exited normally."
if "%DEBUG%"=="1" pause
popd & endlocal & exit /b 0

:fail
call :log "[FAIL] Exit code: %ERRORLEVEL%"
start "" notepad "%LOGFILE%"
echo (Press any key to close this window...)
pause >nul
popd & endlocal & exit /b 1

:log
setlocal
set "MSG=%~1"
echo %MSG%
>> "%LOGFILE%" echo %MSG%
endlocal & exit /b 0
