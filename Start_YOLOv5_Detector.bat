@echo off
echo Starting Coral Reef Health Detection (YOLOv5 Version)...
cd /d "%~dp0YOLOv5_Version"
"%~dp0.venv\Scripts\python.exe" yolov5_detector.py
pause
