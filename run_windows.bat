@echo off
REM One-click run on Windows.
REM   run_windows.bat                 -> 16 GB RAM machine, countries one at a time
REM   run_windows.bat --jobs 3        -> 32 GB+ RAM, 3 countries in parallel (faster)
REM   run_windows.bat --from train    -> resume from a stage (see README)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python -m pip install -r requirements.txt || goto :error
cd src
python -u run_all.py %* || goto :error
cd ..
echo.
echo DONE. Upload this file: %~dp0output\matching_results.tsv
pause
exit /b 0
:error
echo.
echo *** A step failed - scroll up for the error message. ***
pause
exit /b 1
