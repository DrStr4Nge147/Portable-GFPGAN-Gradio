@echo off
setlocal enabledelayedexpansion

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"

REM Define paths relative to the script directory
set "PYTHON_EXE=%SCRIPT_DIR%system\python\python.exe"
set "PIP_SCRIPT_PATH=%SCRIPT_DIR%system\python\Scripts\pip.exe"
set "REQS_FILE=%SCRIPT_DIR%app\requirements.txt"
set "APP_SCRIPT=%SCRIPT_DIR%app\app.py"

REM --- Set local cache dirs ---
set "PYTHONUSERBASE=%SCRIPT_DIR%cache\python_userbase"
set "PIP_CACHE_DIR=%SCRIPT_DIR%cache\pip_cache"
set "TORCH_HOME=%SCRIPT_DIR%cache\torch_cache"

REM -- Add marker files --
set "REQS_MARKER=%SCRIPT_DIR%cache\requirements_installed.flag"
set "PYTORCH_MARKER=%SCRIPT_DIR%cache\pytorch_installed.flag"

mkdir "%PYTHONUSERBASE%" >nul 2>nul
mkdir "%PIP_CACHE_DIR%" >nul 2>nul
mkdir "%TORCH_HOME%" >nul 2>nul

REM --- Detect GPU Info ---
echo ============================================================
echo GFPGAN Portable Launcher
echo ============================================================

set "GPU_NAME="
for /f "skip=1 tokens=*" %%G in ('wmic path win32_VideoController get Name') do (
    if not defined GPU_NAME (
        set "GPU_NAME=%%G"
        goto :AfterGPUDetect
    )
)

:AfterGPUDetect
echo GPU Detected: %GPU_NAME%

REM --- Map GPU to compatible CUDA version ---
set "CUDA_SUPPORTED=0"
set "CUDA_VERSION="
set "TORCH_VERSION=1.12.1"
set "TORCHVISION_VERSION=0.13.1"
set "TORCHAUDIO_VERSION=0.12.1"

echo Checking CUDA compatibility...

echo %GPU_NAME% | findstr /i "RTX 20" >nul && (
    set CUDA_VERSION=cu116
    set CUDA_SUPPORTED=1
)
echo %GPU_NAME% | findstr /i "RTX 30" >nul && (
    set CUDA_VERSION=cu117
    set TORCH_VERSION=1.13.1
    set TORCHVISION_VERSION=0.14.1
    set TORCHAUDIO_VERSION=0.13.1
    set CUDA_SUPPORTED=1
)
echo %GPU_NAME% | findstr /i "RTX 40" >nul && (
    set CUDA_VERSION=cu118
    set TORCH_VERSION=2.0.1
    set TORCHVISION_VERSION=0.15.2
    set TORCHAUDIO_VERSION=2.0.2
    set CUDA_SUPPORTED=1
)
echo %GPU_NAME% | findstr /i "GTX 16" >nul && (
    set CUDA_VERSION=cu116
    set CUDA_SUPPORTED=1
)

if "%CUDA_SUPPORTED%"=="1" (
    set PYTORCH_CUDA_URL=https://download.pytorch.org/whl/%CUDA_VERSION%
    echo Compatible GPU detected. Will use CUDA: %CUDA_VERSION%
) else (
    echo No supported GPU detected. Will use CPU mode.
)

REM --- Check Python ---
echo.
echo Checking Python installation...
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found at %PYTHON_EXE%
    pause
    exit /b 1
)
if not exist "%PIP_SCRIPT_PATH%" (
    echo Pip not found. Installing pip...
    "%PYTHON_EXE%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo ERROR: Pip installation failed.
        pause
        exit /b 1
    )
)
echo Python found: %PYTHON_EXE%
echo.

REM --- Check PyTorch Installation ---
echo Checking PyTorch installation...
set INSTALL_TORCH=0
set FOUND_VERSION=N/A
set FOUND_CUDA=N/A

"%PYTHON_EXE%" -m pip show torch >nul 2>nul
if errorlevel 1 (
    echo PyTorch not found.
    set INSTALL_TORCH=1
) else (
    "%PYTHON_EXE%" -c "import torch; print(f'Torch={torch.__version__};CUDA={torch.cuda.is_available()}')" > pytorch_check.tmp 2>&1
    if exist pytorch_check.tmp (
        for /f "tokens=1,2 delims=;" %%a in (pytorch_check.tmp) do (
            for /f "tokens=1,2 delims==" %%i in ("%%a") do if /i "%%i"=="Torch" set FOUND_VERSION=%%j
            for /f "tokens=1,2 delims==" %%i in ("%%b") do if /i "%%i"=="CUDA" set FOUND_CUDA=%%j
        )
        del pytorch_check.tmp

        echo Found PyTorch version: !FOUND_VERSION!, CUDA Available: !FOUND_CUDA!

        if /i "!FOUND_CUDA!"=="True" (
            if "!FOUND_VERSION!"=="%TORCH_VERSION%" (
                echo PyTorch is compatible with CUDA.
            ) else (
                echo PyTorch version mismatch. Reinstalling...
                set INSTALL_TORCH=1
            )
        ) else (
            echo CUDA not available. Reinstalling with correct version...
            set INSTALL_TORCH=1
        )
    ) else (
        echo Failed to inspect PyTorch. Will reinstall.
        set INSTALL_TORCH=1
    )
)

if "%INSTALL_TORCH%"=="1" (
    del "%PYTORCH_MARKER%" >nul 2>nul
    echo Installing PyTorch...
    if "%CUDA_SUPPORTED%"=="1" (
        "%PYTHON_EXE%" -m pip install torch==%TORCH_VERSION%+%CUDA_VERSION% torchvision==%TORCHVISION_VERSION%+%CUDA_VERSION% torchaudio==%TORCHAUDIO_VERSION% --extra-index-url %PYTORCH_CUDA_URL% --no-cache-dir --user
        if errorlevel 1 (
            echo CUDA installation failed. Falling back to CPU-only...
            "%PYTHON_EXE%" -m pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% --no-cache-dir --user
        )
    ) else (
        "%PYTHON_EXE%" -m pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% --no-cache-dir --user
    )
    echo. > "%PYTORCH_MARKER%"
)

echo.

REM --- Install App Requirements ---
if exist "%REQS_MARKER%" (
    echo App requirements already installed. Skipping.
) else (
    echo Installing app requirements...
    if not exist "%REQS_FILE%" (
        echo ERROR: requirements.txt not found at %REQS_FILE%
        pause
        exit /b 1
    )
    "%PYTHON_EXE%" -m pip install filterpy --use-pep517
    "%PYTHON_EXE%" -m pip install -r "%REQS_FILE%" --no-cache-dir --user
    if errorlevel 1 (
        echo ERROR: Failed to install app requirements.
        pause
        exit /b 1
    )
    echo. > "%REQS_MARKER%"
    echo App requirements installed successfully.
)

echo.
echo ============================================================
echo Launching GFPGAN Application...
echo Access it at http://127.0.0.1:7860
echo ============================================================
echo.

"%PYTHON_EXE%" "%APP_SCRIPT%"

echo.
echo Application finished or closed.
pause
exit /b 0
