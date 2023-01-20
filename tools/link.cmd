@REM Make soft links to post-process tools
@ECHO OFF
SETLOCAL

SET RESR_HOME=D:\tools\realesrgan-ncnn-vulkan
SET RIFE_HOME=D:\tools\rife-ncnn-vulkan
SET FFMPEG_HOME=D:\tools\ffmpeg

@ECHO ON

MKLINK /J %~dp0realesrgan-ncnn-vulkan %RESR_HOME%
MKLINK /J %~dp0rife-ncnn-vulkan       %RIFE_HOME%
MKLINK /J %~dp0ffmpeg                 %FFMPEG_HOME%

ECHO ^>^> Done!
ECHO.

PAUSE
