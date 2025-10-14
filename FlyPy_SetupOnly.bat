
@echo off
setlocal EnableExtensions EnableDelayedExpansion
title FlyPy Setup Only (smart skip)
chcp 65001 >nul
set "ROOT=%~dp0"
cd /d "%ROOT%"
set "LOG_DIR=%ROOT%logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1
set "LOG_FILE=%LOG_DIR%\FlyPy_Setup.log"
break > "%LOG_FILE%"
set "VENV_DIR=%ROOT%.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PY%" (
  for /f "tokens=*" %%V in ('py -3.10 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%V"
  if not defined PYEXE for /f "tokens=*" %%V in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%V"
  if not defined PYEXE ( echo No Python 3.10 found. & exit /b 1 )
  "%PYEXE%" -m venv "%VENV_DIR%" >> "%LOG_FILE%" 2>&1 || (echo venv failed & exit /b 1)
)
set "PYEXE=%VENV_PY%"
"%PYEXE%" -m pip install --upgrade pip --disable-pip-version-check >> "%LOG_FILE%" 2>&1

set "REQS=%ROOT%requirements.txt"
set "LOCK=%ROOT%requirements.lock.txt"
set "STATE=%LOG_DIR%\last_requirements.md5"
set "FLAG=%LOG_DIR%\install_ok.flag"
set "CURMD5=NONE"
if exist "%REQS%" for /f "tokens=1,2" %%A in ('certutil -hashfile "%REQS%" MD5 ^| find /i /v "MD5" ^| find /i /v "certutil"') do set "CURMD5=%%A"
set "OLDMD5=NONE"
if exist "%STATE%" set /p OLDMD5=<"%STATE%"
set "NEED_INSTALL=0"
"%PYEXE%" - <<PYCODE >> "%LOG_FILE%" 2>&1
import importlib.util, sys, subprocess
oks = {m: importlib.util.find_spec(m) is not None for m in ("cv2","PyQt5","psychopy")}
rc = subprocess.call([sys.executable,"-m","pip","check","--disable-pip-version-check"])
print("Probe:", oks, "pip_check_rc:", rc)
sys.exit(0 if all(oks.values()) and rc==0 else 1)
PYCODE
if errorlevel 1 set "NEED_INSTALL=1"
if "%CURMD5%" NEQ "%OLDMD5%" set "NEED_INSTALL=1"
if not exist "%FLAG%" set "NEED_INSTALL=1"

if "%NEED_INSTALL%"=="1" (
  if exist "%REQS%" (
    "%PYEXE%" -m pip install -r "%REQS%" --upgrade-strategy only-if-needed --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || set "PIPFAIL=1"
  ) else if exist "%LOCK%" (
    "%PYEXE%" -m pip install -r "%LOCK%" --upgrade-strategy only-if-needed --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || set "PIPFAIL=1"
  ) else (
    echo No requirements files found. & exit /b 1
  )
  if defined PIPFAIL (
    if exist "%REQS%" (
      "%PYEXE%" -m pip install -r "%REQS%" --upgrade --force-reinstall --no-cache-dir --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || exit /b 1
    ) else (
      "%PYEXE%" -m pip install -r "%LOCK%" --upgrade --force-reinstall --no-cache-dir --disable-pip-version-check >> "%LOG_FILE%" 2>&1 || exit /b 1
    )
  )
  "%PYEXE%" -m pip check --disable-pip-version-check >> "%LOG_FILE%" 2>&1
  >"%STATE%" echo %CURMD5%
  >"%FLAG%" echo ok
) else (
  echo Dependencies OK; skipped installs. >> "%LOG_FILE%"
)
echo Setup complete. Log: "%LOG_FILE%"
endlocal & exit /b 0
