"""
Coral Reef Health Detection (RT-DETR Version)
==============================================
Interactive tool for detecting coral diseases using a trained RT-DETR model.
Supports: Image analysis, Video analysis, and Live webcam stream.

Usage:
    py rtdetr_detector.py
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
from PyQt5.QtWidgets import QApplication, QFileDialog

try:
    from ultralytics import RTDETR
except ImportError:
    print("\n❌ Error: The 'ultralytics' package is required for RT-DETR.")
    print("   Please run: py -m pip install ultralytics")
    sys.exit(1)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "rtdetr_best.pt"
RESULTS_DIR = SCRIPT_DIR / "results"
CONFIDENCE_THRESHOLD = 0.25

# Disease class colors (BGR for OpenCV)
CLASS_COLORS = {
    "Band disease":      (0, 0, 255),      # Red
    "Bleached disease":  (255, 255, 255),   # White
    "Dead Coral":        (128, 128, 128),   # Gray
    "Healthy Coral":     (0, 200, 0),       # Green
    "White Pox Disease": (0, 165, 255),     # Orange
}
DEFAULT_COLOR = (255, 0, 255)  # Magenta fallback


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
def load_model():
    """Load the trained RT-DETR model."""
    if not MODEL_PATH.exists():
        print(f"\n❌ RT-DETR Model file not found: {MODEL_PATH}")
        print("   Make sure you download your trained 'rtdetr_best.pt' from Google Drive")
        print("   and place it in this folder before running.")
        sys.exit(1)

    print("\n🔄 Loading RT-DETR model...")
    model = RTDETR(str(MODEL_PATH))
    print("✅ Model loaded successfully!\n")
    return model


# ─────────────────────────────────────────────
# Drawing Utilities
# ─────────────────────────────────────────────
def get_color(class_name):
    """Get the color for a disease class."""
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def draw_detections(frame, results):
    """Draw bounding boxes and labels on the frame. Returns pure list for summary."""
    detections = []
    
    # Ultralytics results format
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]

            if conf < CONFIDENCE_THRESHOLD:
                continue

            color = get_color(cls_name)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{cls_name} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)

            # Draw label text
            text_color = (0, 0, 0) if cls_name in ("Bleached disease", "Healthy Coral") else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

            detections.append((cls_name, conf))

    return detections


def print_detection_summary(detections, source_name=""):
    """Print a formatted summary of detections to the terminal."""
    if not detections:
        print(f"\n   No coral issues detected in {source_name}.")
        return

    print(f"\n{'─' * 50}")
    print(f"   Detection Results — {source_name}")
    print(f"{'─' * 50}")

    class_counts = Counter(d[0] for d in detections)
    class_max_conf = {}
    for cls_name, conf in detections:
        class_max_conf[cls_name] = max(class_max_conf.get(cls_name, 0), conf)

    for cls_name, count in class_counts.most_common():
        icon = "🟢" if cls_name == "Healthy Coral" else "🔴"
        conf = class_max_conf[cls_name]
        print(f"   {icon} {cls_name}: {count} detection(s), max confidence: {conf:.0%}")

    print(f"   Total detections: {len(detections)}")
    print(f"{'─' * 50}\n")


# ─────────────────────────────────────────────
# Option 1: Image Analysis
# ─────────────────────────────────────────────
def analyze_image(model):
    """Open file dialog, analyze a coral image, display results."""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select a Coral Image",
        "",
        "Image files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;All files (*.*)"
    )

    if not file_path:
        print("\n   No image selected. Returning to menu.")
        return

    print(f"\n🔍 Analyzing: {Path(file_path).name}")

    frame = cv2.imread(file_path)
    if frame is None:
        print(f"   ❌ Could not read image: {file_path}")
        return

    # Run RT-DETR inference
    results = model(frame, verbose=False)
    detections = draw_detections(frame, results)
    print_detection_summary(detections, Path(file_path).name)

    # Save result
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"result_{timestamp}_{Path(file_path).stem}.jpg"
    save_path = RESULTS_DIR / save_name
    cv2.imwrite(str(save_path), frame)
    print(f"   💾 Saved annotated image to: {save_path}")

    # Display
    display = resize_for_display(frame)
    cv2.imshow("RT-DETR Coral Detection", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Option 2: Video Analysis
# ─────────────────────────────────────────────
def analyze_video(model):
    """Open file dialog, analyze a video file frame-by-frame."""
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select a Coral Video",
        "",
        "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*.*)"
    )

    if not file_path:
        print("\n   No video selected. Returning to menu.")
        return

    print(f"\n🎬 Processing video: {Path(file_path).name}")
    print("   Press 'q' on the video window to stop.\n")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return

    frame_count = 0
    total_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t1 = time.time()

        results = model(frame, verbose=False)
        detections = draw_detections(frame, results)
        total_detections.extend(detections)

        fps = 1.0 / (time.time() - t1 + 1e-9)
        cv2.putText(frame, f"FPS: {fps:.1f}  Frame: {frame_count}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        display = resize_for_display(frame)
        cv2.imshow("RT-DETR Video Analysis - Press 'q' to quit", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print_detection_summary(total_detections, Path(file_path).name)


# ─────────────────────────────────────────────
# Option 3: Live Webcam Stream
# ─────────────────────────────────────────────
def live_stream(model):
    """Open webcam for real-time RT-DETR detection."""
    print("\n📷 Starting live webcam stream...")
    print("   Press 'q' on the video window to stop.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   ❌ Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t1 = time.time()
        results = model(frame, verbose=False)
        draw_detections(frame, results)

        fps = 1.0 / (time.time() - t1 + 1e-9)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("RT-DETR Live Stream - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Display Utility
# ─────────────────────────────────────────────
def resize_for_display(frame, max_width=1280, max_height=720):
    """Resize frame to fit on screen while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width and h <= max_height:
        return frame
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


# ─────────────────────────────────────────────
# Main Menu
# ─────────────────────────────────────────────
def show_menu():
    """Display the interactive menu."""
    print("\n" + "=" * 50)
    print("   🪸 RT-DETR CORAL HEALTH DETECTOR 🪸")
    print("=" * 50)
    print()
    print("   [1]  Select Image")
    print("   [2]  Select Video")
    print("   [3]  Live Video Stream")
    print("   [0]  Exit")
    print()
    print("-" * 50)


def main():
    print("\n" + "=" * 50)
    print("   RT-DETR CORAL REEF HEALTH DETECTION SYSTEM")
    print("   Powered by Ultralytics Transformers")
    print("=" * 50)

    model = load_model()

    while True:
        show_menu()
        choice = input("   Enter your choice (1/2/3/0): ").strip()

        if choice == "1":
            analyze_image(model)
        elif choice == "2":
            analyze_video(model)
        elif choice == "3":
            live_stream(model)
        elif choice == "0":
            print("\n   Goodbye! Thank you for protecting coral reefs.\n")
            break
        else:
            print("\n   Invalid choice. Please enter 1, 2, 3, or 0.")


if __name__ == "__main__":
    main()
