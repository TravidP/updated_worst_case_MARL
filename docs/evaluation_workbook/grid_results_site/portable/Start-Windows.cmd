@echo off
setlocal
cd /d "%~dp0"
set "CBWCE_PROGRAM=%~dp0Start-CBWCE.exe"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "CBWCE_PROGRAM=%~dp0bin\windows-arm64\cbwce-viewer.exe"
if /I "%PROCESSOR_ARCHITEW6432%"=="ARM64" set "CBWCE_PROGRAM=%~dp0bin\windows-arm64\cbwce-viewer.exe"
"%CBWCE_PROGRAM%" %*
if errorlevel 1 pause
