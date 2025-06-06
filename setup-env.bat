@echo off
REM ------------------------------------------------------------
REM setup_env.bat
REM 
REM This script activates (or creates + activates) a Python 3 virtual 
REM environment in the “venv” subfolder and installs dependencies 
REM listed in requirements.txt.
REM 
REM Usage: Double-click this file or run from a Command Prompt:
REM    setup_env.bat
REM ------------------------------------------------------------

REM Python3.10 required for Tensorflow on Windows
REM pyenv local 3.10

REM 1) Switch to the directory where this script resides
cd /d "%~dp0"

REM 2) If “venv” does not exist, create it using python
if not exist "venv" (
    echo Creating virtual environment in "venv" with python...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Aborting.
        exit /b 1
    )

    REM ─── Pause for 5 seconds to let venv finish writing files ───
    echo Waiting 5 seconds for venv to finish setting up...
    timeout /t 5 /nobreak >nul
)

REM 3) Activate the virtual environment
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Could not activate virtual environment. Aborting.
    exit /b 1
)

REM 4) (Optional) Upgrade pip inside the venv
python -m pip install --upgrade pip

REM 5) Install dependencies from requirements.txt
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install one or more packages. Check the errors above.
        exit /b 1
    )
) else (
    echo requirements.txt not found. Please add your dependencies there.
)

echo.
echo All done. The virtual environment is active, and dependencies are installed.
echo To deactivate, run: deactivate
