@echo off
setlocal enabledelayedexpansion

title PORTABLE GFPGAN Process (Keep this open to use the app)
REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM Define paths relative to the script directory
set "PYTHON_EXE=%SCRIPT_DIR%\system\python\python.exe"
set "LAUNCHER_SCRIPT=%SCRIPT_DIR%\launcher.py"

REM --- Detect GPU Info ---
echo ============================================================
echo GFPGAN Portable Launcher (Batch Initializer)
echo ============================================================

set "GPU_NAME=Unknown"
for /f "skip=1 tokens=*" %%G in ('wmic path win32_VideoController get Name 2^>nul') do (
    if not "%%G"=="" (
        if "!GPU_NAME!"=="Unknown" (
            set "RAW_GPU_NAME=%%G"
            for /f "tokens=* delims= " %%H in ("!RAW_GPU_NAME!") do set "GPU_NAME=%%H"
            echo Raw WMIC Output Line Used: !RAW_GPU_NAME!
            echo Cleaned GPU Name for Matching: !GPU_NAME!
        )
    )
)

if "%GPU_NAME%"=="Unknown" (
   echo Warning: Could not detect GPU using WMIC. Assuming CPU mode.
) else (
   echo GPU Detected: %GPU_NAME%
)

REM --- Map GPU to compatible CUDA version ---
set "CUDA_SUPPORTED=0"
set "CUDA_SUFFIX="
set "PYTORCH_CUDA_URL="
set "TORCH_VERSION=2.0.1"
set "TORCHVISION_VERSION=0.15.2"
set "TORCHAUDIO_VERSION=2.0.2"

echo Checking CUDA compatibility...
if not "%GPU_NAME%"=="Unknown" (
    echo !GPU_NAME! | findstr /i /c:"RTX 40" >nul && (
        set CUDA_SUFFIX=+cu118
        set TORCH_VERSION=2.0.1
        set TORCHVISION_VERSION=0.15.2
        set TORCHAUDIO_VERSION=2.0.2
        set CUDA_SUPPORTED=1
    )
    if !CUDA_SUPPORTED!==0 echo !GPU_NAME! | findstr /i /c:"RTX 30" >nul && (
        set CUDA_SUFFIX=+cu117
        set TORCH_VERSION=1.13.1
        set TORCHVISION_VERSION=0.14.1
        set TORCHAUDIO_VERSION=0.13.1
        set CUDA_SUPPORTED=1
    )
    if !CUDA_SUPPORTED!==0 echo !GPU_NAME! | findstr /i /c:"RTX 20" >nul && (
        set CUDA_SUFFIX=+cu116
        set TORCH_VERSION=1.12.1
        set TORCHVISION_VERSION=0.13.1
        set TORCHAUDIO_VERSION=0.12.1
        set CUDA_SUPPORTED=1
    )
    if !CUDA_SUPPORTED!==0 echo !GPU_NAME! | findstr /i /c:"GTX 16" >nul && (
        set CUDA_SUFFIX=+cu116
        set TORCH_VERSION=1.12.1
        set TORCHVISION_VERSION=0.13.1
        set TORCHAUDIO_VERSION=0.12.1
        set CUDA_SUPPORTED=1
    )
)

if "%CUDA_SUPPORTED%"=="1" (
    if "%CUDA_SUFFIX%"=="+cu116" set PYTORCH_CUDA_URL=https://download.pytorch.org/whl/cu116
    if "%CUDA_SUFFIX%"=="+cu117" set PYTORCH_CUDA_URL=https://download.pytorch.org/whl/cu117
    if "%CUDA_SUFFIX%"=="+cu118" set PYTORCH_CUDA_URL=https://download.pytorch.org/whl/cu118
    echo Compatible GPU detected. Will attempt CUDA install via Python launcher.
    echo Target Suffix: %CUDA_SUFFIX%
    echo Target Versions: Torch=%TORCH_VERSION%, Vision=%TORCHVISION_VERSION%, Audio=%TORCHAUDIO_VERSION%
    echo Extra Index URL: %PYTORCH_CUDA_URL%
) else (
    echo No supported GPU detected or detection failed. Will use CPU mode via Python launcher.
    set CUDA_SUFFIX=""
    set PYTORCH_CUDA_URL=""
    echo Target Versions: Torch=%TORCH_VERSION%, Vision=%TORCHVISION_VERSION%, Audio=%TORCHAUDIO_VERSION% (CPU)
)

echo.
echo Checking Python installation...
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found at "%PYTHON_EXE%"
    pause
    exit /b 1
)
echo Python found: "%PYTHON_EXE%"
echo.

echo Launching Python environment setup and application launcher...
echo This may take a while on the first run for downloads/installs.
echo.

"%PYTHON_EXE%" "%LAUNCHER_SCRIPT%" ^
    --cuda-suffix %CUDA_SUFFIX% ^
    --torch-version %TORCH_VERSION% ^
    --torchvision-version %TORCHVISION_VERSION% ^
    --torchaudio-version %TORCHAUDIO_VERSION% ^
    --index-url %PYTORCH_CUDA_URL%

set LAUNCH_ERRORLEVEL=%ERRORLEVEL%

echo.
echo Batch script finished execution. ErrorLevel: %LAUNCH_ERRORLEVEL%
pause
exit /b %LAUNCH_ERRORLEVEL%