# F1-Winner-predicting-model
F1 Winner Predictor is a lightweight, end-to-end project that predicts Top-20 finishing drivers for a user-specified Grand Prix using historical F1 data. The app combines classical machine learning with a modern, premium Flask front end to deliver an attractive, interactive prediction experience.
F1 Winner Predictor is a full-stack machine-learning powered web application that predicts the Top-20 finishing drivers of any selected Formula 1 Grand Prix.
The system analyzes complete historical F1 datasets (1950–present) and uses a blend of classical ML models to estimate:

🥇 Winning probability

⏱️ Predicted average lap time

🔁 Estimated number of laps

🏎️ Driver ranking for that race

The app features a premium Mercedes-inspired interface, including:

Neon teal theme

Interactive circular probability rings

A rotating Mercedes logo loading screen

Dynamic parallax background using an F1 Mercedes car

Smooth transitions + ultra-modern UI

🧠 Machine Learning Models

Your dataset is used to train three independent models:

XGBoost (multi-class softprob)

Random Forest Classifier

K-Nearest Neighbors

Two regressors:

RF for lap time prediction

RF for lap count prediction

A scoring mechanism selects the Best Model Automatically
(based on Top-1 accuracy during validation).

All models + encoders + scaler are saved under:

models/
   ├── best_model.pkl
   ├── knn_clf.pkl
   ├── rf_clf.pkl
   ├── xgb_clf.pkl
   ├── lap_rf.pkl
   ├── laps_rf.pkl
   ├── scaler.pkl
   └── encoders.pkl

🏗️ Tech Stack
Backend

Python

Flask

Scikit-learn

XGBoost

Pandas, NumPy

Joblib

Frontend

HTML5

CSS3 (Neon Mercedes Theme)

JS interactions

SVG circular progress animation

📁 Project Structure
F1-Predictor/
│
├── app.py                 # Flask server
├── train_all_models.py    # Full auto-training pipeline
│
├── data/                  # Raw .csv datasets
│
├── models/                # Saved machine learning models
│
├── templates/
│     ├── index.html       # Homepage UI
│     ├── results.html     # Results output UI
│
├── static/
│     ├── style.css        # Mercedes neon theme
│     ├── mercedes_logo.png
│     ├── mercedes_car_bg.jpg
│     └── scripts.js       # UI animations (optional)
│
└── README.md

⚙️ Installation & Setup
1. Clone repository
git clone https://github.com/YOUR_USERNAME/F1-Winner-Predictor.git
cd F1-Winner-Predictor

2. Create a virtual environment
python -m venv .venv

3. Activate

Windows:

.venv\Scripts\activate


Mac/Linux:

source .venv/bin/activate

4. Install dependencies
pip install -r requirements.txt

5. Add datasets

Place all your CSVs inside:

data/

6. Train ML models
python train_all_models.py

7. Run the web app
python app.py


Visit → http://127.0.0.1:5000/
