@echo off
setlocal ENABLEDELAYEDEXPANSION

echo.
echo ===========================================
echo  FlyPy - Install Dependencies
echo ===========================================
echo.

REM -- Move to repo root (where this script lives)
cd /d "%~dp0"

REM -- Create venv if missing (prefer py launcher)
if not exist ".venv" (
    echo [INFO] Creating virtual environment (.venv) ...
    where py >NUL 2>&1
    if %ERRORLEVEL%==0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)

REM -- Activate venv
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [ERROR] Could not activate .venv\Scripts\activate.bat
    echo        Make sure Python 3 is installed and accessible.
    pause
    exit /b 1
)

REM -- Auto-create a minimal requirements.txt if missing
if not exist "requirements.txt" (
    echo [WARN] requirements.txt not found. Creating a minimal one ...
    > "requirements.txt" (
        echo # ============================================
        echo # FlyPy Requirements (core runtime)
        echo # ============================================
        echo.
        echo numpy^>=1.26
        echo opencv-python^>=4.8
        echo PyQt5^>=5.15.9
        echo pyserial^>=3.5
        echo.
        echo # PsychoPy for the looming stimulus (install only on Python ^< 3.11)
        echo psychopy^>=2023.2.3^; python_version ^< "3.11"
    )
)

echo [INFO] Upgrading pip, setuptools, and wheel ...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :pip_fail

echo [INFO] Installing from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 goto :pip_fail

echo [INFO] Writing exact versions to requirements.lock.txt ...
pip freeze --exclude-editable > requirements.lock.txt

echo.
echo [SUCCESS] Environment is ready.
echo          Activate later with:  call .venv\Scripts\activate.bat
echo.
goto :eof

:pip_fail
echo.
echo [FAIL] Package installation failed. See messages above.
echo        You may need to run this as Administrator or fix network/proxy settings.
pause
exit /b 1
