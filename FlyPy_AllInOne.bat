
@echo off
setlocal EnableExtensions EnableDelayedExpansion
title FlyPy All-In-One (smart skip installer)
chcp 65001 >nul

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "LOG_DIR=%ROOT%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1
set "LOG_FILE=%LOG_DIR%\FlyPy_Launch.log"
break > "%LOG_FILE%"

call :log "==============================================="
call :log " FlyPy All-In-One (smart skip)"
call :log " Log: %LOG_FILE%"
call :log " Started: %DATE% %TIME%"
call :log "==============================================="

set "VENV_DIR=%ROOT%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

REM ---- detect python / create venv ----
if not exist "%VENV_PY%" (
  for /f "tokens=*" %%V in ('py -3.10 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%V"
  if not defined PYEXE for /f "tokens=*" %%V in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%V"
  if not defined PYEXE (
    call :log "[ERROR] No Python found. Install 3.10."
    goto :fail
  )
  call :log "[Setup] Creating venv → %VENV_DIR%"
  "%PYEXE%" -m venv "%VENV_DIR%" >> "%LOG_FILE%" 2>&1 || (call :log "[ERROR] venv creation failed" & goto :fail)
)

set "PYEXE=%VENV_PY%"
call :log "[Info ] Python: %PYEXE%"
"%PYEXE%" -V >> "%LOG_FILE%" 2>&1
call :log "[Info ] Pip: "
"%PYEXE%" -m pip --version >> "%LOG_FILE%" 2>&1

REM ---- Only upgrade pip (avoid setuptools churn) ----
call :log "[Setup] Upgrading pip (only)"
"%PYEXE%" -m pip install --upgrade pip --disable-pip-version-check >> "%LOG_FILE%" 2>&1

REM ---- health + hash check ----
set "REQS=%ROOT%requirements.txt"
set "LOCK=%ROOT%requirements.lock.txt"
set "STATE=%LOG_DIR%\last_requirements.md5"
set "FLAG=%LOG_DIR%\install_ok.flag"

set "CURMD5=NONE"
if exist "%REQS%" (
  for /f "tokens=1,2" %%A in ('certutil -hashfile "%REQS%" MD5 ^| find /i /v "MD5" ^| find /i /v "certutil"') do set "CURMD5=%%A"
)
set "OLDMD5=NONE"
if exist "%STATE%" (
  set /p OLDMD5=<"%STATE%"
)

call :log "[Check] requirements.txt MD5: %CURMD5% (prev: %OLDMD5%)"

REM Quick import probe (no heavy imports) + pip check
set "NEED_INSTALL=0"
"%PYEXE%" - <<PYCODE > "%LOG_FILE%" 2>&1
import importlib.util, sys, subprocess
def ok(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False
oks = {"cv2": ok("cv2"), "PyQt5": ok("PyQt5"), "psychopy": ok("psychopy")}
print("[Probe] imports:", oks)
rc = subprocess.call([sys.executable,"-m","pip","check","--disable-pip-version-check"])
print("[Probe] pip check rc:", rc)
sys.exit(0 if all(oks.values()) and rc==0 else 1)
PYCODE
if errorlevel 1 set "NEED_INSTALL=1"

if "%CURMD5%" NEQ "%OLDMD5%" set "NEED_INSTALL=1"
if not exist "%FLAG%" set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="1" (
  call :log "[Install] Installing dependencies (only-if-needed)"
  if exist "%REQS%" (
    "%PYEXE%" -m pip install -r "%REQS%" --upgrade-strategy only-if-needed --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || set "PIPFAIL=1"
  ) else if exist "%LOCK%" (
    "%PYEXE%" -m pip install -r "%LOCK%" --upgrade-strategy only-if-needed --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || set "PIPFAIL=1"
  ) else (
    call :log "[ERROR] No requirements files found."
    goto :fail
  )
  if defined PIPFAIL (
    call :log "[Repair] Re-installing forcibly from requirements.txt/lock"
    if exist "%REQS%" (
      "%PYEXE%" -m pip install -r "%REQS%" --upgrade --force-reinstall --no-cache-dir --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || goto :fail
    ) else (
      "%PYEXE%" -m pip install -r "%LOCK%" --upgrade --force-reinstall --no-cache-dir --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || goto :fail
    )
  )
  REM post-check
  "%PYEXE%" -m pip check --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || (call :log "[WARN ] pip check has warnings")
  >"%STATE%" echo %CURMD5%
  >"%FLAG%" echo ok
) else (
  call :log "[Skip ] Dependencies look good; skipping installs."
)

REM ---- Spinnaker DLL PATH hints ----
for %%S in (
  "C:\Program Files\Teledyne FLIR\Spinnaker\bin64"
  "C:\Program Files\Teledyne FLIR\Spinnaker\lib64"
  "C:\Program Files\FLIR Systems\Spinnaker\bin64"
  "C:\Program Files\FLIR Systems\Spinnaker\lib64"
) do (
  if exist %%~S (
    echo !PATH! | find /I "%%~S" >nul || set "PATH=%%~S;!PATH!"
  )
)

REM ---- Run app ----
if exist "%ROOT%FlyAPI.py" (
  call :log "[Run  ] Launching FlyAPI.py"
  "%PYEXE%" "%ROOT%FlyAPI.py" %* >> "%LOG_FILE%" 2>&1
  set "APP_RC=%ERRORLEVEL%"
) else if exist "%ROOT%FlyAPI.exe" (
  call :log "[Run  ] Launching FlyAPI.exe"
  "%ROOT%FlyAPI.exe" %* >> "%LOG_FILE%" 2>&1
  set "APP_RC=%ERRORLEVEL%"
) else (
  call :log "[ERROR] FlyAPI.py / FlyAPI.exe not found"
  goto :fail
)

if not "%APP_RC%"=="0" (
  call :log "[ERROR] FlyAPI exited with %APP_RC%"
  goto :fail
)

call :log "[OK   ] Done."
goto :success

:fail
  echo.
  echo FAILED. See log: "%LOG_FILE%"
  start "" notepad "%LOG_FILE%"
  pause >nul
  endlocal & exit /b 1

:success
  call :log "Success. Log at: %LOG_FILE%"
  endlocal & exit /b 0

:log
  setlocal EnableDelayedExpansion
  set "MSG=%~1"
  if not defined MSG set "MSG="
  echo.!MSG!
  >> "%LOG_FILE%" echo.!MSG!
  endlocal & exit /b 0
