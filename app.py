from flask import Flask, render_template, request, url_for
from pathlib import Path
from main import process_image, YOLO

app = Flask(__name__)

UPLOAD_FOLDER = Path("static/uploads")
OUTPUT_FOLDER = Path("static/outputs")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

yolo_model = YOLO("yolov8s.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("images")
        results = []
        for file in files:
            save_path = UPLOAD_FOLDER / file.filename
            file.save(save_path)
            rec = process_image(save_path, yolo_model)
            if rec is not None:
                rec["processed_image"] = url_for("static", filename=f"outputs/{rec['processed_image']}")
                results.append(rec)
        return render_template("results.html", results=results)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
