@echo off
setlocal
set DRIVER=%~dp0..\drivers\CH341SER.EXE
if not exist "%DRIVER%" (
  echo CH340/CH341 driver not found at %DRIVER%
  echo Please download from the official WCH page and place it there:
  echo   https://www.wch-ic.com/downloads/CH341SER_EXE.html
  echo   https://www.wch.cn/downloads/CH341SER_EXE.html
  pause
  exit /b 1
)
echo Launching CH340/CH341 driver installer...
start "" "%DRIVER%"
