"""
Coral Reef Health Detection Application
========================================
Interactive tool for detecting coral diseases using a trained YOLOv5 model.
Supports: Image analysis, Video analysis, and Live webcam stream.

Usage:
    py coral_detector.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add YOLOv5 engine to path (needed for torch.load to find models.yolo)
YOLOV5_DIR = Path(__file__).parent / "yolov5_engine"
if YOLOV5_DIR.exists():
    sys.path.insert(0, str(YOLOV5_DIR))

import cv2
import torch
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog

# Windows compatibility fix: The model was trained on Linux (Colab) and saved with PosixPaths.
# When torch.load() unpickles it on Windows, it crashes. We mock PosixPath to WindowsPath.
if sys.platform == 'win32':
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "yolov5_best.pt"
RESULTS_DIR = SCRIPT_DIR / "results"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

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
# Model Loading (offline, no internet needed)
# ─────────────────────────────────────────────
def load_model():
    """Load the trained YOLOv5 model from best.pt using raw torch.load."""
    if not MODEL_PATH.exists():
        print(f"\n❌ Model file not found: {MODEL_PATH}")
        print("   Make sure 'best.pt' is in the same folder as this script.")
        sys.exit(1)

    print("\n🔄 Loading coral detection model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    # Load the checkpoint (contains full model architecture + weights)
    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model = ckpt['model'].float().eval()

    # Extract class names
    if hasattr(model, 'names'):
        if isinstance(model.names, dict):
            names = model.names
        elif isinstance(model.names, list):
            names = {i: n for i, n in enumerate(model.names)}
        else:
            names = {0: 'class0'}
    else:
        names = {0: 'Band disease', 1: 'Bleached disease', 2: 'Dead Coral',
                 3: 'Healthy Coral', 4: 'White Pox Disease'}

    # Get stride
    stride = int(model.stride.max()) if hasattr(model, 'stride') else 32

    print(f"   Classes: {list(names.values())}")
    print("✅ Model loaded successfully!\n")
    return model, names, stride, device


# ─────────────────────────────────────────────
# Preprocessing & Inference
# ─────────────────────────────────────────────
def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize and pad image to square while maintaining aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Perform NMS on inference results."""
    # Filter by confidence
    xc = prediction[..., 4] > conf_thres
    output = []

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            output.append(torch.zeros((0, 6)))
            continue

        # Compute class confidence = obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]

        # Box (center x, center y, w, h) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Best class per detection
        conf, cls_idx = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, cls_idx.float()), 1)[conf.view(-1) > conf_thres]

        if not x.shape[0]:
            output.append(torch.zeros((0, 6)))
            continue

        # NMS per class
        boxes = x[:, :4]
        scores = x[:, 4]
        classes = x[:, 5]

        # Offset boxes by class for per-class NMS
        offset = classes * 4096
        boxes_offset = boxes + offset[:, None]

        from torchvision.ops import nms
        keep = nms(boxes_offset, scores, iou_thres)
        output.append(x[keep])

    return output


def run_inference(model, frame, names, stride, device):
    """Run YOLOv5 inference on a single frame. Returns list of (class_name, conf, x1, y1, x2, y2)."""
    orig_shape = frame.shape[:2]

    # Preprocess
    img, ratio, (dw, dh) = letterbox(frame, IMG_SIZE)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]

    # NMS
    detections_raw = non_max_suppression(pred, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

    results = []
    for det in detections_raw:
        if det is not None and len(det):
            # Scale boxes back to original image
            det[:, 0] = (det[:, 0] - dw) / ratio
            det[:, 1] = (det[:, 1] - dh) / ratio
            det[:, 2] = (det[:, 2] - dw) / ratio
            det[:, 3] = (det[:, 3] - dh) / ratio

            # Clip to image bounds
            det[:, 0].clamp_(0, orig_shape[1])
            det[:, 1].clamp_(0, orig_shape[0])
            det[:, 2].clamp_(0, orig_shape[1])
            det[:, 3].clamp_(0, orig_shape[0])

            for *xyxy, conf, cls_id in det:
                cls_name = names.get(int(cls_id), f'class{int(cls_id)}')
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                results.append((cls_name, float(conf), x1, y1, x2, y2))

    return results


# ─────────────────────────────────────────────
# Drawing Utilities
# ─────────────────────────────────────────────
def get_color(class_name):
    """Get the color for a disease class."""
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for cls_name, conf, x1, y1, x2, y2 in detections:
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
    for cls_name, conf, *_ in detections:
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
def analyze_image(model, names, stride, device):
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

    # Read image
    frame = cv2.imread(file_path)
    if frame is None:
        print(f"   ❌ Could not read image: {file_path}")
        return

    # Run detection
    detections = run_inference(model, frame, names, stride, device)
    draw_detections(frame, detections)
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
    cv2.imshow("Coral Disease Detection - Press any key to close", display)
    print("   Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Option 2: Video Analysis
# ─────────────────────────────────────────────
def analyze_video(model, names, stride, device):
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
        print(f"   ❌ Could not open video: {file_path}")
        return

    frame_count = 0
    total_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t1 = time.time()

        detections = run_inference(model, frame, names, stride, device)
        draw_detections(frame, detections)
        total_detections.extend(detections)

        # FPS overlay
        fps = 1.0 / (time.time() - t1 + 1e-9)
        cv2.putText(frame, f"FPS: {fps:.1f}  Frame: {frame_count}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        display = resize_for_display(frame)
        cv2.imshow("Coral Video Analysis - Press 'q' to quit", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print_detection_summary(total_detections, Path(file_path).name)


# ─────────────────────────────────────────────
# Option 3: Live Webcam Stream
# ─────────────────────────────────────────────
def live_stream(model, names, stride, device):
    """Open webcam for real-time coral disease detection."""
    print("\n📷 Starting live webcam stream...")
    print("   Press 'q' on the video window to stop.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   ❌ Could not open webcam. Make sure a camera is connected.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("   ❌ Failed to read from webcam.")
            break

        t1 = time.time()

        detections = run_inference(model, frame, names, stride, device)
        draw_detections(frame, detections)

        # FPS overlay
        fps = 1.0 / (time.time() - t1 + 1e-9)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Coral Live Detection - Press 'q' to quit", frame)

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
    print("   CORAL REEF HEALTH DETECTION SYSTEM")
    print("=" * 50)
    print()
    print("   [1]  Select Image")
    print("   [2]  Select Video")
    print("   [3]  Live Video Stream")
    print("   [0]  Exit")
    print()
    print("-" * 50)


def main():
    """Main application entry point."""
    print("\n" + "=" * 50)
    print("   CORAL REEF HEALTH DETECTION SYSTEM")
    print("   Powered by YOLOv5 + Custom Trained Model")
    print("=" * 50)

    # Load model once
    model, names, stride, device = load_model()

    while True:
        show_menu()
        choice = input("   Enter your choice (1/2/3/0): ").strip()

        if choice == "1":
            analyze_image(model, names, stride, device)
        elif choice == "2":
            analyze_video(model, names, stride, device)
        elif choice == "3":
            live_stream(model, names, stride, device)
        elif choice == "0":
            print("\n   Goodbye! Thank you for protecting coral reefs.\n")
            break
        else:
            print("\n   Invalid choice. Please enter 1, 2, 3, or 0.")


if __name__ == "__main__":
    main()
