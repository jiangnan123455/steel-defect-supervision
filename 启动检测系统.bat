@echo off
title Steel Defect Detection System
cd /d "%~dp0"

rem If the service is already listening on 8501, just open the browser
netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo  System is already running. Opening browser...
    start "" http://localhost:8501
    exit /b 0
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo.
    echo [ERROR] Python environment not found: .venv\Scripts\streamlit.exe
    echo.
    echo Please run these commands in this folder first:
    echo   1. python -m venv .venv
    echo   2. .venv\Scripts\activate ^&^& pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "weights\best.pt" (
    echo.
    echo [WARN] weights\best.pt not found. The web page will show a
    echo        model-missing hint. Train first: python train.py --resume
    echo.
)

rem Skip Streamlit first-run welcome prompt (it blocks non-interactive start)
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    mkdir "%USERPROFILE%\.streamlit" 2>nul
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo showWelcome = false
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo ==============================================
echo   Steel Defect Detection System
echo   YOLO + Supervision + Streamlit
echo ==============================================
echo.
echo  Starting server, please wait...
echo  The browser will open automatically.
echo  If it does not, open the URL shown below.
echo.
echo  Stop: close this window or press Ctrl+C
echo.

call ".venv\Scripts\streamlit.exe" run app.py --browser.gatherUsageStats false

echo.
echo Server stopped.
pause
