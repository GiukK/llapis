@echo off
REM build-check -- can this machine build and run LLapis? (cli\build-check.ps1)
REM   build-check            check the prerequisites
REM   build-check -Build     ...then configure, build, test
REM   build-check -Smoke     ...then start the server and ask one question
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0cli\build-check.ps1" %*
exit /b %ERRORLEVEL%
