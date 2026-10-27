@echo off
rem GHOST 2.0 full head swap (CUDA). Usage:
rem   start_headswap.bat <source photo> <target image or video> <output file> [extra options]
rem Extra options: --start 5 --end 15 (seconds), --temporal-blend 0.2, --no-keep-source-colors, --paste rect
if "%~3"=="" (
	echo Usage: start_headswap.bat ^<source photo^> ^<target image or video^> ^<output file^> [extra options]
	pause
	exit /b 1
)
call conda run -p D:\nsfw\envs\ghost2 --no-capture-output python D:\nsfw\facefusion\ghost-2.0\headswap\ghost_headswap.py --source "%~f1" --target "%~f2" --output "%~f3" %4 %5 %6 %7 %8 %9
pause
