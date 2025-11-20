# app.py
from flask import Flask, render_template, request, send_from_directory
import os
from infer_predictions import infer, load_all  # infer returns rows, model_name
import infer_predictions as ip

app = Flask(__name__)
ROOT = os.path.abspath(os.path.dirname(__file__))

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    gp = request.form.get("grand_prix", "").strip()
    date = request.form.get("date", "").strip()
    weather = request.form.get("weather", "").strip()
    model_choice = request.form.get("model", "best")
    rows, used_model = infer(gp, date, weather, model_choice)
    # rows is list of dicts as described
    return render_template("results.html", rows=rows, gp=gp, date=date, weather=weather, model=used_model)

# static files served by Flask automatically if using app.static_folder = "static"
if __name__ == "__main__":
    print("Starting app")
    app.run(debug=True, host="0.0.0.0", port=5000)
