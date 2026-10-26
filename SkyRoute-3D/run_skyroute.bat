@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0vendor\pyglet\__init__.py" (
  python -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo Could not install the graphics library.
    pause
    exit /b 1
  )
)
python skyroute3d.py
if errorlevel 1 (
  echo.
  echo SkyRoute could not start. Check that Python 3.10 or newer is installed.
  pause
)
endlocal
