@REM Test script for post-process pipeline
@ECHO OFF
SETLOCAL

REM =================================================
REM Real-ESRGAN model ckpt
SET RESR_MODEL=realesr-animevideov3
REM image upscale rate, must choose from [2, 3, 4]
SET RESR_UPSCALE=2
REM RIFE model ckpt
SET RIFE_MODEL=rife-v4
REM interpolated frame count, set 0 to simply Nx2
SET RIFE_INTERP=0
REM rendered video fps, higher value requires more interpolations
SET FPS=20
REM =================================================


IF NOT EXIST "%~1" (
  ECHO Usage: %~nx0 path/to/^<image_folder^> [-k]
  ECHO    -k     keep cache data for debug
  PAUSE
  GOTO :EOF
)

SET RESR_HOME=%~dp0tools\realesrgan-ncnn-vulkan
SET RIFE_HOME=%~dp0tools\rife-ncnn-vulkan
SET FFMPEG_HOME=%~dp0tools\ffmpeg

SET RESR_BIN=realesrgan-ncnn-vulkan.exe
SET RIFE_BIN=rife-ncnn-vulkan.exe
SET FFMPEG_BIN=ffmpeg.exe

PATH %RESR_HOME%;%PATH%
PATH %RIFE_HOME%;%PATH%
PATH %FFMPEG_HOME%\bin;%FFMPEG_HOME%;%PATH%

SET IMAGE_FOLDER=%~1
SET RESR_FOLDER="%IMAGE_FOLDER%\resr"
SET RIFE_FOLDER="%IMAGE_FOLDER%\rife"
SET OUT_FILE="%IMAGE_FOLDER%\synth.mp4"

ECHO ==================================================

ECHO [1/5] image super-resolution
IF EXIST %RESR_FOLDER% GOTO skip_resr
MKDIR %RESR_FOLDER%
%RESR_BIN% -v -s %RESR_UPSCALE% -n %RESR_MODEL% -i %IMAGE_FOLDER% -o %RESR_FOLDER%
IF ERRORLEVEL 1 GOTO die
:skip_resr
ECHO ==================================================

ECHO [2/5] video frame-interpolation
IF EXIST %RIFE_FOLDER% GOTO skip_rife
MKDIR %RIFE_FOLDER%
SET NFRAMES=%RESR_FOLDER%

%RIFE_BIN% -v -n %RIFE_INTERP% -m %RIFE_MODEL% -i %RESR_FOLDER% -o %RIFE_FOLDER%
IF ERRORLEVEL 1 GOTO die
:skip_rife
ECHO ==================================================

ECHO [3/5] render video
%FFMPEG_BIN% -y -framerate %FPS% -i %RIFE_FOLDER%\%%08d.png -crf 20 -c:v libx264 -pix_fmt yuv420p %OUT_FILE%
IF ERRORLEVEL 1 GOTO die
ECHO ==================================================

IF NOT "%2"=="-k" (
  RMDIR /S /Q %RESR_FOLDER%
  RMDIR /S /Q %RIFE_FOLDER%
)

explorer.exe /e,/select,%OUT_FILE%

ECHO ^>^> Done!
ECHO.
GOTO :eof

:die
ECHO ^<^< errorlevel: %ERRORLEVEL%

:eof
