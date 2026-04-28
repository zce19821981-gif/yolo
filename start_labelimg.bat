@echo off
set SCRIPT_DIR=%~dp0
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
  set PYTHON_BIN=%SCRIPT_DIR%.venv\Scripts\python.exe
) else if exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
  set PYTHON_BIN=%SCRIPT_DIR%venv\Scripts\python.exe
) else (
  set PYTHON_BIN=python
)

set IMAGE_DIR=%SCRIPT_DIR%data
set LABEL_DIR=%SCRIPT_DIR%labels

if not "%~1"=="" (
  if not "%~2"=="" (
    set IMAGE_DIR=%~1
    set LABEL_DIR=%~2
  ) else (
    set IMAGE_DIR=%SCRIPT_DIR%data\%~1
    set LABEL_DIR=%SCRIPT_DIR%labels\%~1
  )
)

"%PYTHON_BIN%" -m labelImg.labelImg "%IMAGE_DIR%" "%SCRIPT_DIR%configs\labelimg_classes.txt" "%LABEL_DIR%"
