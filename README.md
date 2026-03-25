# 🪸 Coral Reef Health Detection System

An AI-powered desktop application to detect coral reef diseases using two powerful models:
- **YOLOv5** — Fast and reliable detection
- **RT-DETR** — Advanced Transformer-based detection with higher precision

---

## 📋 Before You Start — What You Need

You need to install **two free programs** on your laptop before anything else.

### 1. Install Python 3.11
1. Go to: https://www.python.org/downloads/release/python-3119/
2. Scroll to the bottom and click **"Windows installer (64-bit)"**
3. Run the downloaded file
4. ✅ **VERY IMPORTANT:** On the first screen, tick the box that says **"Add Python to PATH"** before clicking Install
5. Click **Install Now** and wait for it to finish

### 2. Install Git
1. Go to: https://git-scm.com/download/win
2. Click the first download link
3. Run the downloaded file and click **Next** on every screen (all defaults are fine)
4. Click **Install** and wait for it to finish

### 3. Install uv (the fast package manager)
1. Press `Windows Key + R`, type `cmd`, press Enter
2. A black window will appear. Type this command and press Enter:
```
pip install uv
```
3. Wait for it to finish (you'll see text scrolling)

---

## 📥 Step 1 — Download the Project

1. Press `Windows Key + R`, type `cmd`, press Enter
2. In the black window, type this command exactly and press Enter:
```
git clone https://github.com/praneetha-t/Coral.git
```
3. Wait for it to finish. A new folder called `Coral` will appear wherever your terminal was open.
4. Now navigate into the folder:
```
cd Coral
```

---

## ⚙️ Step 2 — Set Up the Environment

This creates an isolated, safe Python workspace just for this project. Copy and paste each command one at a time and press Enter after each one:

```
uv venv --python 3.11
```
```
uv pip install ultralytics opencv-python numpy==1.26.4 PyQt5 pandas seaborn tqdm torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
```

⏳ This will take about 5-10 minutes as it downloads libraries. Please be patient!

---

## 🧠 Step 3 — Download the AI Model Files

The trained AI brain files are too large to store on GitHub. You need to download them separately.

### YOLOv5 Model:
1. Download the file `yolov5_best.pt` from the link your instructor provided
2. Place it inside the `YOLOv5_Version` folder

### RT-DETR Model:
1. Download the file `rtdetr_best.pt` from the link your instructor provided
2. Place it inside the `RT_DETR_Version` folder

Your final folder structure should look like this:
```
Coral/
├── YOLOv5_Version/
│   ├── yolov5_best.pt       ← Place here
│   ├── yolov5_detector.py
│   └── Inference_YOLOv5.ipynb
├── RT_DETR_Version/
│   ├── rtdetr_best.pt       ← Place here
│   ├── rtdetr_detector.py
│   └── Inference_RT_DETR.ipynb
├── Start_YOLOv5_Detector.bat
└── Start_RT_DETR_Detector.bat
```

---

## ▶️ Step 4 — Run the Application

### Option A: YOLOv5 Detection (Fast)
Simply open your `Coral` folder in File Explorer and **double-click** `Start_YOLOv5_Detector.bat`

### Option B: RT-DETR Detection (More Precise)
Simply open your `Coral` folder in File Explorer and **double-click** `Start_RT_DETR_Detector.bat`

A black window will open showing a menu like this:
```
==================================================
   CORAL REEF HEALTH DETECTION SYSTEM
==================================================

   [1]  Select Image
   [2]  Select Video
   [3]  Live Video Stream
   [0]  Exit
```

- Press `1` and Enter → A file picker will open. Select your coral image.
- Press `2` and Enter → A file picker will open. Select your coral video.
- Press `3` and Enter → Opens your webcam for live detection.
- Press `0` and Enter → Closes the program.

---

## ☁️ Option C: Run in Google Colab (No Installation Needed!)

If the above steps feel too complicated, you can run everything online for free!

### For YOLOv5 on Colab:
1. Go to https://colab.research.google.com
2. Click **File** → **Upload notebook**
3. Upload `YOLOv5_Version/Inference_YOLOv5.ipynb`
4. Upload `yolov5_best.pt` to the Colab sidebar (left panel, Files icon)
5. Click **Runtime** → **Run All**
6. When Step 2 prompts you, click **Choose Files** and select your coral image

### For RT-DETR on Colab:
Same steps as above, but use `RT_DETR_Version/Inference_RT_DETR.ipynb` and `rtdetr_best.pt`

---

## 🔍 Understanding the Results

When the app detects something, the terminal will show:
```
──────────────────────────────────────────────────
   Detection Results — your_image.jpg
──────────────────────────────────────────────────
   🔴 Band disease: 3 detection(s), max confidence: 87%
   🔴 Bleached disease: 1 detection(s), max confidence: 74%
   🟢 Healthy Coral: 2 detection(s), max confidence: 91%
   Total detections: 6
──────────────────────────────────────────────────
```

A popup window will also display the image with colored boxes drawn around each detected area. The annotated image is automatically saved to the `results/` folder.

### Disease Color Guide:
| Color | Meaning |
|-------|---------|
| 🔴 Red box | Band Disease |
| ⬜ White box | Bleached Coral |
| 🩶 Gray box | Dead Coral |
| 🟢 Green box | Healthy Coral |
| 🟠 Orange box | White Pox Disease |

---

## ❓ Common Problems & Fixes

**Problem:** Double-clicking the .bat file shows an error about a missing `.pt` file
**Fix:** You forgot to place the model file in the correct folder. See Step 3 above.

**Problem:** The black window says `ModuleNotFoundError`
**Fix:** The environment setup in Step 2 did not complete fully. Open `cmd`, navigate to the Coral folder with `cd Coral` and re-run the `uv pip install ...` command from Step 2.

**Problem:** The image window does not appear
**Fix:** Press any key on the black terminal window to bring focus back.

---

*Built with YOLOv5 + RT-DETR | Powered by Ultralytics | Coral Reef Conservation Project*
