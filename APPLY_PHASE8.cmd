@echo off
setlocal
if not exist "crypto_trading" (
  echo Run this script from the repository root.
  exit /b 2
)
for /f "delims=" %%F in ('dir /s /b "%~dp0crypto_trading\*" "%~dp0scripts\*" "%~dp0tests\*" "%~dp0data_persistence\*" 2^>nul') do (
  set "SRC=%%F"
  call set "REL=%%SRC:%~dp0=%%"
  call if not exist "%%~dpREL" mkdir "%%~dpREL"
  call copy /y "%%SRC%%" "%%REL%%" >nul
)
echo Phase 8 files copied. Run the commands in EXECUTION_CHECKLIST.md.
exit /b 0
