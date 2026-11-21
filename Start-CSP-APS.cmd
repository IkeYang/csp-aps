@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Install the Windows environment as described in README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m cspaps.cli gui %*
exit /b %errorlevel%
