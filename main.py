import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Optional
from uuid import uuid4
from datetime import datetime
import os

# allowed file extensions used in app.py
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# Output folder default; can be overridden by caller
DEFAULT_OUT_DIR = Path("static/outputs")
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_yolo_model(model_path: str = "yolov8s.pt") -> YOLO:
    """
    Load and return a YOLO model instance.
    """
    return YOLO(model_path)


# -----------------------------
# Helper functions
# -----------------------------
def map_measurement_to_score(value: float, min_val: float, max_val: float) -> int:
    if value <= min_val:
        return 1
    if value >= max_val:
        return 9
    score = 1 + (value - min_val) * 8.0 / (max_val - min_val)
    return int(round(score))


def compute_measurements_from_extremes(left, right, top, bottom):
    """
    Compute body length (px), height (px) and rump angle (deg) from extreme points.
    left, right, top, bottom are (x, y) tuples.
    """
    body_length_px = np.linalg.norm(np.array(right) - np.array(left))
    height_px = np.linalg.norm(np.array(bottom) - np.array(top))

    # rump angle proxy: angle between vectors (right - top) and (bottom - right)
    v1 = np.array(right) - np.array(top)
    v2 = np.array(bottom) - np.array(right)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cosang = np.dot(v1, v2) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cosang)))
    return float(body_length_px), float(height_px), angle_deg


def normalize_measurements(body_px: float, height_px: float, img_shape) -> (float, float):
    h, w = img_shape[:2]
    diag = np.sqrt(w ** 2 + h ** 2)
    if diag <= 0:
        return 0.0, 0.0
    return body_px / diag, height_px / diag


def interpret_score(score: int, kind: str = "body") -> str:
    if kind == "body":
        if score <= 3:
            return "Underweight"
        elif score <= 6:
            return "Average"
        else:
            return "Good"
    if kind == "height":
        if score <= 3:
            return "Short"
        elif score <= 6:
            return "Medium"
        else:
            return "Tall"
    if kind == "rump":
        if score <= 3:
            return "Low Rump"
        elif score <= 6:
            return "Average Rump"
        else:
            return "High Rump"
    if kind == "total":
        if score <= 10:
            return "Poor"
        elif score <= 20:
            return "Average"
        else:
            return "Excellent"
    return "Unknown"


def draw_overlay(image: np.ndarray, mask: np.ndarray, extremes: dict, scores: dict, out_path: Path) -> np.ndarray:
    vis = image.copy()
    if mask is not None:
        colored_mask = np.zeros_like(vis)
        colored_mask[mask > 0] = (0, 200, 0)
        vis = cv2.addWeighted(vis, 0.7, colored_mask, 0.3, 0)
    for name, pt in extremes.items():
        cv2.circle(vis, tuple(pt), 8, (255, 0, 0), -1)
        cv2.putText(vis, name, (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    text = f"Body:{scores['body']} Height:{scores['height']} Rump:{scores['rump']} Total:{scores['total']}"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), vis)
    return vis


def _extract_mask_from_crop(crop_rgb: np.ndarray, x_offset: int, y_offset: int, full_shape) -> Optional[np.ndarray]:
    """
    Given a cropped RGB patch, produce a binary mask in the full-image coordinates.
    Returns None on failure.
    """
    try:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        # shift contour coordinates back to full image coords
        c[:, :, 0] += x_offset
        c[:, :, 1] += y_offset
        mask = np.zeros((full_shape[0], full_shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 1, -1)
        return mask
    except Exception:
        return None


def process_image(img_path: Path, yolo_model: YOLO, output_dir: Path = DEFAULT_OUT_DIR) -> Optional[List[Dict]]:
    """
    Process a single image file and return a list of result dicts (one per detected animal).
    Each dict contains scores, conditions and the filename of the processed annotated image.
    Returns None or empty list on failure/no detections.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("[ERROR] Could not read image:", img_path)
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = []
    try:
        yolo_results = yolo_model(str(img_path))
    except Exception as e:
        print("[WARN] YOLO processing failed:", e)
        return None

    # iterate detections and process each relevant class
    detection_found = False
    for r in yolo_results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                cls = int(box.cls[0])
                label = yolo_model.names.get(cls, str(cls)).lower()
                if label not in {"cow", "cattle", "ox", "bull", "buffalo"}:
                    continue

                detection_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_rgb.shape[1] - 1, x2), min(img_rgb.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img_rgb[y1:y2, x1:x2]
                mask = _extract_mask_from_crop(crop, x1, y1, img_rgb.shape)
                if mask is None:
                    continue

                contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                c = max(contours, key=cv2.contourArea)
                left = tuple(c[c[:, :, 0].argmin()][0])
                right = tuple(c[c[:, :, 0].argmax()][0])
                top = tuple(c[c[:, :, 1].argmin()][0])
                bottom = tuple(c[c[:, :, 1].argmax()][0])

                body_len_px, height_px, rump_angle_deg = compute_measurements_from_extremes(left, right, top, bottom)
                norm_body, norm_height = normalize_measurements(body_len_px, height_px, img_rgb.shape)
                score_body = map_measurement_to_score(norm_body, 0.15, 0.55)
                score_height = map_measurement_to_score(norm_height, 0.08, 0.38)
                score_rump = map_measurement_to_score(rump_angle_deg, 15, 110)
                total_score = score_body + score_height + score_rump

                extremes = {"left": left, "right": right, "top": top, "bottom": bottom}
                scores = {"body": score_body, "height": score_height, "rump": score_rump, "total": total_score}

                # unique output filename
                out_filename = f"processed_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex}_{Path(img_path).name}"
                out_path = Path(output_dir) / out_filename
                draw_overlay(img_rgb.copy(), mask, extremes, scores, out_path)

                result = {
                    "score_body": score_body,
                    "score_height": score_height,
                    "score_rump": score_rump,
                    "total_score": total_score,
                    "body_condition": interpret_score(score_body, "body"),
                    "height_condition": interpret_score(score_height, "height"),
                    "rump_condition": interpret_score(score_rump, "rump"),
                    "total_condition": interpret_score(total_score, "total"),
                    "processed_image": out_filename,
                    "bbox": (x1, y1, x2, y2),
                    "label": label
                }
                results.append(result)
            except Exception as exc:
                print("[WARN] detection processing error:", exc)
                continue

    if not detection_found:
        return []

    return results