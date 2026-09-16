@echo off
setlocal

title ECG Arrhythmia Classifier - Launcher

echo.
echo  ========================================================
echo   ECG Arrhythmia Classifier - Local Launcher
echo  ========================================================
echo.

:: ── Paths ───────────────────────────────────────────────────
set "BACKEND=%~dp0backend"
set "FRONTEND=%~dp0frontend"

:: ────────────────────────────────────────────────────────────
:: 1. Find Python
:: ────────────────────────────────────────────────────────────
echo [1/4] Checking Python...
where py >nul 2>&1
if not errorlevel 1 (
    set "PY=py"
    goto python_ok
)
where python >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
    goto python_ok
)
echo.
echo  ERROR: Python not found.
echo  Install Python 3.10+ from https://www.python.org/downloads/
echo  Ensure it is on your system PATH.
echo.
echo  Press any key to exit...
pause >nul
exit /b 1

:python_ok
echo        Python found.

:: ────────────────────────────────────────────────────────────
:: 2. Find Node / npm
:: ────────────────────────────────────────────────────────────
echo [2/4] Checking Node.js...
where npm >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: npm not found.
    echo  Install Node.js v18+ from https://nodejs.org/
    echo.
    echo  Press any key to exit...
    pause >nul
    exit /b 1
)
echo        npm found.

:: ────────────────────────────────────────────────────────────
:: 3. Backend Python deps
:: ────────────────────────────────────────────────────────────
echo [3/4] Checking backend Python packages...
%PY% -c "import fastapi, uvicorn, torch, sklearn, pandas" >nul 2>&1
if not errorlevel 1 (
    echo        All packages already installed.
    goto frontend_check
)
echo        Installing core packages...
%PY% -m pip install fastapi "uvicorn[standard]" numpy scikit-learn pandas joblib python-multipart pydantic aiofiles
%PY% -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo        Installing PyTorch CPU (~200 MB, please wait...^)
    %PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu
)
echo        Backend packages ready.

:: ────────────────────────────────────────────────────────────
:: 4. Frontend npm deps
:: ────────────────────────────────────────────────────────────
:frontend_check
echo [4/4] Checking frontend packages...
if exist "%FRONTEND%\node_modules" (
    echo        node_modules already present.
) else (
    echo        Running npm install, please wait...
    pushd "%FRONTEND%"
    npm install
    popd
)

:: ────────────────────────────────────────────────────────────
:: Launch Backend
:: ────────────────────────────────────────────────────────────
echo.
echo  Launching Backend  ^(http://localhost:8000^)...
start "ECG Backend [port 8000]" cmd /k "pushd "%BACKEND%" && %PY% -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: Short delay so backend can start loading model before frontend
timeout /t 3 /nobreak >nul

:: ────────────────────────────────────────────────────────────
:: Launch Frontend
:: ────────────────────────────────────────────────────────────
echo  Launching Frontend ^(http://localhost:3000^)...
start "ECG Frontend [port 3000]" cmd /k "pushd "%FRONTEND%" && set BROWSER=none && npm start"

:: ────────────────────────────────────────────────────────────
:: Open browser once servers are ready
:: ────────────────────────────────────────────────────────────
echo.
echo  Waiting 18 seconds for servers to start...
timeout /t 18 /nobreak >nul

start "" "http://localhost:3000"
start "" "http://localhost:8000/docs"

echo.
echo  ========================================================
echo   Both servers are running!
echo.
echo   Frontend  -^>  http://localhost:3000
echo   API Docs  -^>  http://localhost:8000/docs
echo.
echo   To STOP: close the two ECG terminal windows.
echo  ========================================================
echo.
