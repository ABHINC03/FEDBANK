```bat
@echo off
setlocal

echo ==========================================
echo       FedBank Conda Environment Setup
echo ==========================================
echo.

REM Check if Conda is installed
where conda >nul 2>&1

if %errorlevel% neq 0 (
    echo [ERROR] Conda was not found.
    echo.
    echo Please install Anaconda or Miniconda first.
    echo Then run this script again.
    pause
    exit /b 1
)

echo [INFO] Conda detected.
echo.

REM Check if fedbank environment exists
conda env list | findstr /B /C:"fedbank " >nul 2>&1

if %errorlevel%==0 (
    echo [INFO] fedbank environment already exists.
) else (
    echo [INFO] Creating fedbank environment...
    conda create -n fedbank python=3.11 -y

    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to create fedbank environment.
        pause
        exit /b 1
    )

    echo.
    echo [SUCCESS] fedbank environment created.
)

echo.
echo [INFO] Activating fedbank environment...
call conda activate fedbank

echo.
echo ==========================================
echo   FedBank Environment Ready
echo ==========================================
echo.

python --version
conda env list

echo.
echo You are now using the fedbank environment.
echo.

pause
```
