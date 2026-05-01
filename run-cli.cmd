@echo off
REM Force this repo's v3 package if another checkout is earlier on sys.path (e.g. site-packages .pth).
set "PYTHONPATH=%~dp0src"
cd /d "%~dp0"
python -u -m v3.cli %*
exit /b %ERRORLEVEL%
