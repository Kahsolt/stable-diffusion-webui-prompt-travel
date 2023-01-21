@REM Handy script for post-process pipeline
@ECHO OFF
SETLOCAL

TITLE Post-processing for prompt-travel...

REM remeber base path and script name
SET _=%~dp0
SET $=%~nx0
SHIFT

REM init configs or make default
SET CONFIG_FILE=%_%postprocess-config.cmd
IF EXIST %CONFIG_FILE% GOTO skip_init_cfg
COPY %CONFIG_FILE%.example %CONFIG_FILE%
IF ERRORLEVEL 1 GOTO die
:skip_init_cfg

REM load configs
CALL %CONFIG_FILE%
IF ERRORLEVEL 1 GOTO die

REM assert required arguments
IF /I "%~0"=="-c" (
  SET CLEAN_FLAG=1
  SHIFT
)
SET IMAGE_FOLDER=%~0
SHIFT

REM show help
IF NOT EXIST "%IMAGE_FOLDER%" (
  ECHO Usage: %$% [-c] ^<image_folder^> [upscale] [interp] [fps] [resr_model] [rife_model]
  ECHO    -c            clean cache data when done
  ECHO    upscale       image upsampling rate ^(default: %RESR_UPSCALE%^)
  ECHO    interp        interpolated video frame count ^(default: %RIFE_INTERP%^)
  ECHO    fps           rendered video frame rate ^(default: %FPS%^)
  ECHO    resr_model    Real-ESRGAN model checkpoint name ^(default: %RESR_MODEL%^)
  ECHO    rife_model    RIFE model checkpoint name ^(default: %RIFE_MODEL%^)
  ECHO.
  ECHO    e.g. %$% D:\images
  ECHO         %$% -c D:\images
  ECHO         %$% D:\images 2 0
  ECHO         %$% D:\images 4 120 24
  ECHO         %$% D:\images 4 0 24 realesr-animevideov3 rife-v2.3
  ECHO    note:
  ECHO         ^<args^> arguments are required
  ECHO         ^[args^] arguments are optional
  ECHO.
  GOTO :end
)

REM override optional arguments by command line
IF NOT "%~0"=="" (
  SET RESR_UPSCALE=%~0
  SHIFT
)
IF NOT "%~0"=="" (
  SET RIFE_INTERP=%~0
  SHIFT
)
IF NOT "%~0"=="" (
  SET FPS=%~0
  SHIFT
)
IF NOT "%~0"=="" (
  SET RESR_MODEL=%~0
  SHIFT
)
IF NOT "%~0"=="" (
  SET RIFE_MODEL=%~0
  SHIFT
)

REM prepare paths
SET TOOL_HOME=%_%tools
SET RESR_HOME=%TOOL_HOME%\realesrgan-ncnn-vulkan
SET RIFE_HOME=%TOOL_HOME%\rife-ncnn-vulkan
SET FFMPEG_HOME=%TOOL_HOME%\ffmpeg

SET BBOX_BIN=busybox.exe
SET RESR_BIN=realesrgan-ncnn-vulkan.exe
SET RIFE_BIN=rife-ncnn-vulkan.exe
SET FFMPEG_BIN=ffmpeg.exe

PATH %TOOL_HOME%;%PATH%
PATH %RESR_HOME%;%PATH%
PATH %RIFE_HOME%;%PATH%
PATH %FFMPEG_HOME%\bin;%FFMPEG_HOME%;%PATH%

SET RESR_FOLDER=%IMAGE_FOLDER%\resr
SET RIFE_FOLDER=%IMAGE_FOLDER%\rife
SET OUT_FILE=%IMAGE_FOLDER%\synth.mp4

REM show configs for debug
ECHO ==================================================
ECHO RESR_MODEL   = %RESR_MODEL%
ECHO RESR_UPSCALE = %RESR_UPSCALE%
ECHO RIFE_MODEL   = %RIFE_MODEL%
ECHO RIFE_INTERP  = %RIFE_INTERP%
ECHO FPS          = %FPS%
ECHO RESR_FOLDER  = %RESR_FOLDER%
ECHO RIFE_FOLDER  = %RIFE_FOLDER%
ECHO OUT_FILE     = %OUT_FILE%
ECHO.

ECHO ^>^> wait for %WAIT_BEFORE_START% seconds before start...
%BBOX_BIN% sleep %WAIT_BEFORE_START%
IF ERRORLEVEL 1 GOTO die
ECHO ^>^> start processing!

REM start processing
ECHO ==================================================

ECHO [1/3] image super-resolution
IF EXIST %RESR_FOLDER% GOTO skip_resr
MKDIR %RESR_FOLDER%
%RESR_BIN% -v -s %RESR_UPSCALE% -n %RESR_MODEL% -i %IMAGE_FOLDER% -o %RESR_FOLDER%
IF ERRORLEVEL 1 GOTO die
:skip_resr

ECHO ==================================================

ECHO [2/3] video frame-interpolation
IF EXIST %RIFE_FOLDER% GOTO skip_rife
MKDIR %RIFE_FOLDER%
SET NFRAMES=%RESR_FOLDER%

%RIFE_BIN% -v -n %RIFE_INTERP% -m %RIFE_MODEL% -i %RESR_FOLDER% -o %RIFE_FOLDER%
IF ERRORLEVEL 1 GOTO die
:skip_rife

ECHO ==================================================

ECHO [3/3] render video
%FFMPEG_BIN% -y -framerate %FPS% -i %RIFE_FOLDER%\%%08d.png -crf 20 -c:v libx264 -pix_fmt yuv420p %OUT_FILE%
IF ERRORLEVEL 1 GOTO die

ECHO ==================================================

REM clean cache
IF "%CLEAN_FLAG%"=="1" (
  RMDIR /S /Q %RESR_FOLDER%
  RMDIR /S /Q %RIFE_FOLDER%
)

REM finished
ECHO ^>^> file saved to %OUT_FILE%
IF "%EXPLORER_FLAG%"=="1" (
  explorer.exe /e,/select,%OUT_FILE%
)

ECHO ^>^> Done!
ECHO.
GOTO :end

REM error handle
:die
ECHO ^<^< Error!
ECHO ^<^< errorlevel: %ERRORLEVEL%
ECHO.

:end
PAUSE
