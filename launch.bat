@echo off

set PYTHON=venv_stable\Scripts\python.exe
set ERROR_REPORTING=FALSE
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

mkdir tmp 2>NUL

echo Using Python: %PYTHON%
echo Launching SimpleStable with GPU optimizations...

%PYTHON% app.py
if %ERRORLEVEL% == 0 goto :end
goto :show_error

:show_error
echo.
echo Exit code: %errorlevel%
echo Launch unsuccessful. Check the error messages above.
pause
goto :end

:end
if %ERRORLEVEL% == 0 (
    echo SimpleStable launched successfully.
) else (
    echo Launch unsuccessful. Check the error messages above.
    pause
)
