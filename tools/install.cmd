@REM Auto download and setup post-process tools
@ECHO OFF
SETLOCAL

REM Usage: install.cmd        install and keep .downloaded folder
REM        install.cmd -c     install and clean .downloaded folder

TITLE Install tools for post-process...
CD %~dp0

REM paths to web resources
SET CURL_BIN=curl.exe -L -C -

SET BBOX_URL=https://frippery.org/files/busybox/busybox.exe
SET BBOX_BIN=busybox.exe
SET UNZIP_BIN=%BBOX_BIN% unzip

SET RESR_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip
SET RESR_ZIP=realesrgan-ncnn-vulkan.zip
SET RESR_DIR=realesrgan-ncnn-vulkan

SET RIFE_URL=https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip
SET RIFE_ZIP=rife-ncnn-vulkan.zip
SET RIFE_DIR=rife-ncnn-vulkan
SET RIFE_RDIR=rife-ncnn-vulkan-20221029-windows

SET FFMPEG_URL=https://github.com/GyanD/codexffmpeg/releases/download/5.1.2/ffmpeg-5.1.2-full_build-shared.zip
SET FFMPEG_ZIP=ffmpeg.zip
SET FFMPEG_DIR=ffmpeg
SET FFMPEG_RDIR=ffmpeg-5.1.2-full_build-shared

REM make cache tmpdir
SET DOWNLOAD_DIR=.download
IF NOT EXIST %DOWNLOAD_DIR% MKDIR %DOWNLOAD_DIR%
ATTRIB +H %DOWNLOAD_DIR%

REM start installation
ECHO ==================================================

ECHO [0/3] download BusyBox
IF EXIST %BBOX_BIN% GOTO skip_bbox
%CURL_BIN% %BBOX_URL% -o %BBOX_BIN%
:skip_bbox

ECHO ==================================================

ECHO [1/3] install Real-ESRGAN
IF EXIST %RESR_DIR% GOTO skip_resr
IF EXIST %DOWNLOAD_DIR%\%RESR_ZIP% GOTO skip_dl_resr
ECHO ^>^> download from %RESR_URL%
%CURL_BIN% %RESR_URL% -o %DOWNLOAD_DIR%\%RESR_ZIP%
IF ERRORLEVEL 1 GOTO die
:skip_dl_resr
ECHO ^>^> uzip %RESR_ZIP%
MKDIR %RESR_DIR%
%UNZIP_BIN% %DOWNLOAD_DIR%\%RESR_ZIP% -d %RESR_DIR%
IF ERRORLEVEL 1 GOTO die
:skip_resr

ECHO ==================================================

ECHO [2/3] install RIFE
IF EXIST %RIFE_DIR% GOTO skip_rife
IF EXIST %DOWNLOAD_DIR%\%RIFE_ZIP% GOTO skip_dl_rife
ECHO ^>^> download from %RIFE_URL%
%CURL_BIN% %RIFE_URL% -o %DOWNLOAD_DIR%\%RIFE_ZIP%
IF ERRORLEVEL 1 GOTO die
:skip_dl_rife
ECHO ^>^> uzip %RIFE_ZIP%
%UNZIP_BIN% %DOWNLOAD_DIR%\%RIFE_ZIP%
IF ERRORLEVEL 1 GOTO die
RENAME %RIFE_RDIR% %RIFE_DIR%
:skip_rife

ECHO ==================================================

ECHO [3/3] install FFmpeg
IF EXIST %FFMPEG_DIR% GOTO skip_ffmpeg
IF EXIST %DOWNLOAD_DIR%\%FFMPEG_ZIP% GOTO skip_dl_ffmpeg
ECHO ^>^> download from %FFMPEG_URL%
%CURL_BIN% %FFMPEG_URL% -o %DOWNLOAD_DIR%\%FFMPEG_ZIP%
IF ERRORLEVEL 1 GOTO die
:skip_dl_ffmpeg
ECHO ^>^> uzip %FFMPEG_ZIP%
%UNZIP_BIN% %DOWNLOAD_DIR%\%FFMPEG_ZIP%
IF ERRORLEVEL 1 GOTO die
RENAME %FFMPEG_RDIR% %FFMPEG_DIR%
:skip_ffmpeg

ECHO ==================================================

REM clean cache
IF /I "%~1"=="-c" (
  ATTRIB -H %DOWNLOAD_DIR%
  RMDIR /S /Q %DOWNLOAD_DIR%
)

REM finished
ECHO ^>^> Done!
ECHO.
GOTO :end

REM error handle
:die
ECHO ^<^< Error!
ECHO ^<^< errorlevel: %ERRORLEVEL%

:end
PAUSE
