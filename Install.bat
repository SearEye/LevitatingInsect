@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ============================================
REM Install.bat — Bootstrap + Install requirements
REM - If requirements.txt exists: install from it
REM - Else: write a default requirements.txt, then install
REM - Writes requirements.lock.txt (pip freeze)
REM - Uses local .venv
REM ============================================

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

REM Choose Python launcher
where py >nul 2>&1
if %ERRORLEVEL%==0 ( set "PYTHON=py" ) else ( set "PYTHON=python" )

REM Create venv if missing
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

REM If requirements.txt missing, write a sensible default (core + optional commented)
if not exist requirements.txt (
  echo [INFO] Writing default requirements.txt ...
  > requirements.txt (
    echo numpy^>=1.26
    echo opencv-python^>=4.8
    echo PyQt5^>=5.15.9
    echo pyserial^>=3.5
    REM PsychoPy only for Python ^< 3.11 (CMD needs ^< escape)
    echo psychopy^>=2023.2.3; python_version ^< "3.11"

    REM ----- Optional tools (commented; uncomment if needed) -----
    echo # pillow^>=10.0
    echo # imageio^>=2.34
    echo # imageio-ffmpeg^>=0.4
    echo # av^>=10.0
    echo # opencv-contrib-python^>=4.8
    echo # pandas^>=2.2
    echo # pyarrow^>=16.0
    echo # pydantic^>=2.5
    echo # pyyaml^>=6.0.1
    echo # rich^>=13.7
    echo # typer^>=0.12
    echo # numba^>=0.59
    echo # tqdm^>=4.66
    echo # pywin32^>=306; sys_platform == "win32"
    echo # screeninfo^>=0.8
    echo # ruff^>=0.5
    echo # black^>=24.4
    echo # isort^>=5.13
    echo # mypy^>=1.10
    echo # types-PyYAML^>=6.0.12.20240808
    echo # pytest^>=8.2
    echo # pytest-qt^>=4.4
    echo # pytest-cov^>=5.0
    echo # pre-commit^>=3.7
    echo # sphinx^>=7.3
    echo # furo^>=2024.8.6
    echo # sphinx-autodoc-typehints^>=2.1
    echo # myst-parser^>=3.0
    echo # NOTE: FLIR/Spinnaker SDK (PySpin) is vendor-installed, not pip.
  )
) else (
  echo [INFO] Found existing requirements.txt — using it.
)

echo [INFO] Installing from requirements.txt ...
pip install -r requirements.txt
if ERRORLEVEL 1 (
  echo [ERROR] pip install failed. See output above.
  popd & endlocal & exit /b 1
)

echo [INFO] Writing exact versions to requirements.lock.txt ...
pip freeze > requirements.lock.txt

echo [INFO] Quick import check (numpy, cv2, PyQt5, serial) ...
python -c "import importlib; [importlib.import_module(m) for m in ('numpy','cv2','PyQt5','serial')]; print('Core imports OK.')"

echo [SUCCESS] Requirements installed.
popd
endlocal & exit /b 0
