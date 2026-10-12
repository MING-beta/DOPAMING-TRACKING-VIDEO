@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1

set "VIDEO=hockey_game.mp4"
if not "%~1"=="" set "VIDEO=%~1"

if not exist "%VIDEO%" (
  py -3 -c "import os; print('\uc601\uc0c1 \ud30c\uc77c\uc744 \ucc3e\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4: ' + os.environ['VIDEO'])"
  py -3 -c "print('\uc774 \ud3f4\ub354\uc5d0 hockey_game.mp4 \ub97c \ub450\uac70\ub098, \uc601\uc0c1 \ud30c\uc77c\uc744 start.bat \uc704\ub85c \ub04c\uc5b4\ub2e4 \ub193\uc73c\uc138\uc694.')"
  py -3 -c "print('\n\ucc3d\uc744 \ub2eb\uc73c\ub824\uba74 \ud0a4\ub97c \ub204\ub974\uc138\uc694.')"
  pause >nul
  exit /b 1
)

:ask
set "NAME="
set "NUMBER="
py -3 -c "print('\uc785\ub825\ud558\uc9c0 \uc54a\uc73c\uba74 \uad04\ud638 \uc548 \uac12\uc744 \uc0ac\uc6a9\ud569\ub2c8\ub2e4.')"
py -3 -c "print('\uc774\ub984(EAN)')"
set /p "NAME="
if "%NAME%"=="" set "NAME=EAN"
py -3 -c "print('\ubc88\ud638(26)')"
set /p "NUMBER="
if "%NUMBER%"=="" set "NUMBER=26"
echo %NAME%| findstr /r "^[A-Za-z][A-Za-z]*$" >nul
if errorlevel 1 (
  py -3 -c "print('\uc774\ub984\uc740 \uc601\ubb38\uc73c\ub85c \uc785\ub825\ud558\uc138\uc694.')"
  goto ask
)
echo %NUMBER%| findstr /r "^[0-9][0-9]$" >nul
if errorlevel 1 (
  py -3 -c "print('\ub4f1\ubc88\ud638\ub294 \uc22b\uc790 2\uc790\ub9ac\uc5ec\uc57c \ud569\ub2c8\ub2e4.')"
  goto ask
)

py -3 -c "import os; print('\n\ubd84\uc11d\uc744 \uc2dc\uc791\ud569\ub2c8\ub2e4: ' + os.environ['NAME'].upper() + ' ' + os.environ['NUMBER'])"
py -3 track_player.py --video "%VIDEO%" --name "%NAME%" --number "%NUMBER%" --no-parseq --no-sam2
py -3 -c "print('\n\uc885\ub8cc\ud588\uc2b5\ub2c8\ub2e4. \ucc3d\uc744 \ub2eb\uc73c\ub824\uba74 \ud0a4\ub97c \ub204\ub974\uc138\uc694.')"
pause >nul
