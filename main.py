import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Output folder
OUT_DIR = Path("static/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def map_measurement_to_score(value, min_val, max_val):
    if value <= min_val: return 1
    if value >= max_val: return 9
    score = 1 + (value - min_val) * 8.0 / (max_val - min_val)
    return int(round(score))

def compute_measurements_from_extremes(left, right, top, bottom):
    body_length_px = np.linalg.norm(np.array(right) - np.array(left))
    height_px = np.linalg.norm(np.array(bottom) - np.array(top))
    # rump angle proxy
    v1 = np.array(right) - np.array(top)
    v2 = np.array(bottom) - np.array(right)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-8)
    cosang = np.clip(cosang, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosang))
    return body_length_px, height_px, angle_deg

def normalize_measurements(body_px, height_px, img_shape):
    h, w = img_shape[:2]
    diag = np.sqrt(w**2 + h**2)
    return body_px/diag, height_px/diag

def interpret_score(score, kind="body"):
    # Interpretation example
    if kind=="body":
        if score<=3: return "Underweight"
        elif score<=6: return "Average"
        else: return "Good"
    if kind=="height":
        if score<=3: return "Short"
        elif score<=6: return "Medium"
        else: return "Tall"
    if kind=="rump":
        if score<=3: return "Low Rump"
        elif score<=6: return "Average Rump"
        else: return "High Rump"
    if kind=="total":
        if score<=10: return "Poor"
        elif score<=20: return "Average"
        else: return "Excellent"

def draw_overlay(image, mask, extremes, scores, out_path):
    vis = image.copy()
    if mask is not None:
        colored_mask = np.zeros_like(vis)
        colored_mask[mask>0] = (0,200,0)
        vis = cv2.addWeighted(vis, 0.7, colored_mask, 0.3, 0)
    for name, pt in extremes.items():
        cv2.circle(vis, tuple(pt), 8, (255,0,0), -1)
        cv2.putText(vis, name, (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    text = f"Body:{scores['body']} Height:{scores['height']} Rump:{scores['rump']} Total:{scores['total']}"
    cv2.putText(vis, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3)
    cv2.putText(vis, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
    cv2.imwrite(str(out_path), vis)
    return vis

# -----------------------------
# Main image processing
# -----------------------------
def process_image(img_path, yolo_model):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print("[ERROR] Could not read image:", img_path)
        return None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = None

    # YOLO detection
    try:
        results = yolo_model(str(img_path))
        chosen_box = None
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None: continue
            for box in boxes:
                cls = int(box.cls[0])
                label = yolo_model.names.get(cls, str(cls))
                if label.lower() in ["cow", "cattle", "ox", "bull", "buffalo"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    chosen_box = (x1,y1,x2,y2)
                    break
            if chosen_box: break

        if chosen_box:
            x1,y1,x2,y2 = chosen_box
            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((5,5), np.uint8)
            thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                c[:,:,0] += x1
                c[:,:,1] += y1
                cv2.drawContours(mask, [c], -1, 1, -1)
    except Exception as e:
        print("[WARN] YOLO processing failed:", e)
        return None

    if mask is None:
        print("[ERROR] Could not produce mask; skipping image.")
        return None

    contours, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    left = tuple(c[c[:,:,0].argmin()][0])
    right = tuple(c[c[:,:,0].argmax()][0])
    top = tuple(c[c[:,:,1].argmin()][0])
    bottom = tuple(c[c[:,:,1].argmax()][0])

    body_len_px, height_px, rump_angle_deg = compute_measurements_from_extremes(left,right,top,bottom)
    norm_body, norm_height = normalize_measurements(body_len_px, height_px, img.shape)
    score_body = map_measurement_to_score(norm_body, 0.15, 0.55)
    score_height = map_measurement_to_score(norm_height, 0.08, 0.38)
    score_rump = map_measurement_to_score(rump_angle_deg, 15, 110)
    total_score = score_body + score_height + score_rump

    extremes = {"left": left, "right": right, "top": top, "bottom": bottom}
    scores = {"body": score_body, "height": score_height, "rump": score_rump, "total": total_score}

    out_img_path = OUT_DIR / f"processed_{Path(img_path).name}"
    draw_overlay(img.copy(), mask, extremes, scores, out_img_path)

    return {
        "score_body": score_body,
        "score_height": score_height,
        "score_rump": score_rump,
        "total_score": total_score,
        "body_condition": interpret_score(score_body, "body"),
        "height_condition": interpret_score(score_height, "height"),
        "rump_condition": interpret_score(score_rump, "rump"),
        "total_condition": interpret_score(total_score, "total"),
        "processed_image": str(out_img_path.name)
    }
