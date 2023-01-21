@REM Make soft links to post-process tools
@ECHO OFF
SETLOCAL

SET RESR_HOME=D:\tools\realesrgan-ncnn-vulkan
SET RIFE_HOME=D:\tools\rife-ncnn-vulkan
SET FFMPEG_HOME=D:\tools\ffmpeg

@ECHO ON

PUSHD %~dp0
MKLINK /J realesrgan-ncnn-vulkan %RESR_HOME%
MKLINK /J rife-ncnn-vulkan       %RIFE_HOME%
MKLINK /J ffmpeg                 %FFMPEG_HOME%
POPD

ECHO ^>^> Done!
ECHO.

PAUSE
