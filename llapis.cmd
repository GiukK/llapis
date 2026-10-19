@echo off
REM llapis -- the canonical command, callable as plain `llapis ...` from cmd or
REM PowerShell without the .\ prefix and without an execution-policy dance.
REM -NoProfile keeps a user's $PROFILE from changing what the runtime sees.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0llapis.ps1" %*
exit /b %ERRORLEVEL%
