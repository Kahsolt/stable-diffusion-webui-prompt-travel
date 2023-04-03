@REM start webui's python venv
@ECHO OFF

SET SD_PATH=%~dp0\..\..
PUSHD %SD_PATH%
SET SD_PATH=%CD%
POPD

REM SET VENV_PATH=C:\Miniconda3
SET VENV_PATH=%SD_PATH%\venv

SET PATH=%VENV_PATH%\Scripts;%PATH%
SET PY_BIN=python.exe

%PY_BIN% --version > NUL
IF ERRORLEVEL 1 GOTO die

DOSKEY pp=python postprocessor.py

CMD /K activate.bat ^& ECHO VENV_PATH: %VENV_PATH% ^& %PY_BIN% --version

GOTO EOF

:die
ECHO ERRORLEVEL: %ERRORLEVEL%
ECHO PATH: %PATH%
ECHO VENV_PATH: %VENV_PATH%
ECHO Python executables:
WHERE python.exe

PAUSE

:EOF
