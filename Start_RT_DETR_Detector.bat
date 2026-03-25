@echo off
echo Starting Coral Reef Health Detection (RT-DETR Version)...
cd /d "%~dp0RT_DETR_Version"
"%~dp0.venv\Scripts\python.exe" rtdetr_detector.py
pause
