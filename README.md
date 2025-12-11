---

# 🐄 ATC Project: Automated Cattle Body Condition Scoring

Smart India Hackathon Prototype Submission
(Not selected, but fully functional as a proof of concept)

---

## 📌 Overview

ATC Project is a web based system that detects cattle in images, segments the animal, extracts geometric features, computes body condition scores and generates annotated images.

The system performs:

* YOLOv8 based cattle detection
* Threshold segmentation
* Contour extraction
* Keypoint identification
* Calculation of body length, height and a rump angle estimate
* Normalized measurement scoring
* Annotated output generation
* Flask based web interface with batch upload support

---

## ⭐ Features

* Clean Flask interface
* Multiple image upload
* YOLOv8 powered detection
* Contour based geometry extraction
* Body, height, rump and total scoring
* Annotated visual output saved automatically
* Easy to use result display

---

## 🔄 System Pipeline

1. Upload one or more images
2. Detect cattle using YOLOv8
3. Segment the detected region
4. Extract the largest contour
5. Find extreme contour points
6. Compute measurements
7. Normalize and score
8. Produce annotated output

---

## 🧰 Tech Stack

* Python
* Flask
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy
* HTML and CSS

---

## 📁 Project Structure

```
ATC_Project/
  app.py
  main.py
  requirements.txt

  static/
    uploads/
    outputs/

  templates/
    index.html
    results.html
```

---

## ⚙️ Installation and Usage

### Clone the repository

```
git clone https://github.com/biswajyoti-nath/ATC_Project.git
cd ATC_Project
```

### Optional: Create a virtual environment

```
python -m venv venv
```

### Install dependencies

```
pip install -r requirements.txt
```

### Add YOLO weights

Download `yolov8s.pt` or `yolov8n.pt` and place it in the project folder.
Example line in code:

```
yolo_model = YOLO("yolov8s.pt")
```

### Run the application

```
python app.py
```

### Open in browser

```
http://127.0.0.1:5000
```

Upload cattle images and view processed results.

---

## 🧮 Scoring Method

The system computes:

* Normalized body length
* Normalized height
* Rump angle estimate

Scores range from 1 to 9 per measurement.

Total score interpretation:

```
1  to 10   Poor
11 to 20   Average
21 to 27   Excellent
```

These thresholds were created for prototype demonstration and are not veterinary validated.

---

## 🚀 Prototype Status and Future Work

* Train a custom YOLO model for Indian cattle
* Replace threshold segmentation with instance segmentation
* Build a labeled dataset for calibration
* Add dashboards and analytics
* Deploy to cloud for field usage

---

## 👤 Author

**Biswajyoti Nath**
Barak Valley Engineering College
Department of Computer Science and Engineering

---
