@echo off
rem GHOST 2.0 head swap web UI. Open the printed "phone" address on a phone connected to the same Wi-Fi.
rem Options: --port 7860, --host 127.0.0.1 (this PC only)
call conda run -p D:\nsfw\envs\ghost2 --no-capture-output python D:\nsfw\facefusion\ghost-2.0\headswap\web_ui.py %*
pause
