@echo off
REM  -------------------------------------------------
REM  easyrag_run.bat
REM  One‑click setup & launch for EasyRAG (Windows)
REM  -------------------------------------------------

:: -------------------------
:: 1) Create the venv once
:: -------------------------
if not exist "EasyRAG_venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv EasyRAG_venv
)

:: -------------------------
:: 2) Activate the venv
:: -------------------------
call ".\EasyRAG_venv\Scripts\activate.bat"

:: -------------------------
:: 3) Install / upgrade uv
:: -------------------------
echo [SETUP] Ensuring 'uv' is installed...
pip install --upgrade uv

:: -------------------------
:: 4) Install project deps
:: -------------------------
echo [SETUP] Installing requirements with uv...
uv pip install -r requirements.txt || (
    echo [ERROR] Requirements install failed.
    goto :eof
)

:: -------------------------
:: 5) Start the API server
::    in a new window
:: -------------------------
echo [RUN] Launching FastAPI backend...
start "EasyRAG API" cmd /k "uvicorn api:app --port 8000"

:: -------------------------
:: 6) Wait until API is up
:: -------------------------
echo [WAIT] Waiting for http://localhost:8000/ ...
:wait_api
curl -s http://localhost:8000/ >nul 2>&1
if errorlevel 1 (
    timeout /t 2 >nul
    goto wait_api
)

:: -------------------------
:: 7) Launch Streamlit UI
:: -------------------------
echo [RUN] API is live - starting Streamlit UI...
streamlit run frontend.py
