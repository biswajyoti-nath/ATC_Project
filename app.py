from pathlib import Path
from uuid import uuid4
from datetime import datetime
from flask import Flask, render_template, request, url_for, abort
from werkzeug.utils import secure_filename
import os
from main import process_image, load_yolo_model, ALLOWED_EXTENSIONS

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
OUTPUT_FOLDER = BASE_DIR / "static" / "outputs"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Load YOLO model (adjust model_path if you keep weights separately)
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8s.pt")
yolo_model = load_yolo_model(MODEL_PATH)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB per request (adjust if needed)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "images" not in request.files:
            abort(400, "No files part in request")
        files = request.files.getlist("images")
        if not files:
            abort(400, "No files uploaded")

        results = []
        for file in files:
            if file.filename == "" or not allowed_file(file.filename):
                # Skip invalid files but continue processing other uploads
                continue

            filename = secure_filename(file.filename)
            unique_name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid4().hex}_{filename}"
            save_path = Path(app.config["UPLOAD_FOLDER"]) / unique_name
            file.save(save_path)

            # process_image can return multiple results (one per detected animal)
            recs = process_image(save_path, yolo_model, output_dir=Path(app.config["OUTPUT_FOLDER"]))
            if not recs:
                # add an entry noting failure to process this uploaded file
                results.append({
                    "input_filename": filename,
                    "error": "No detections or processing failed"
                })
                continue

            # recs is a list; transform each record for template (url_for)
            for rec in recs:
                rec["processed_image"] = url_for("static", filename=f"outputs/{rec['processed_image']}")
                rec["input_filename"] = filename
                results.append(rec)

        return render_template("results.html", results=results)
    return render_template("index.html")


if __name__ == "__main__":
    # for development only; use gunicorn/uvicorn for production
    app.run(debug=True, host="0.0.0.0", port=5000)