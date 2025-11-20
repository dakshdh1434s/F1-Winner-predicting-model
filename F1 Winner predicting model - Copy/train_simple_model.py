# train_all_models.py
# Robust, from-scratch training script for F1 prediction project.
# Produces models/encoders.pkl, scaler.pkl, knn_clf.pkl, rf_clf.pkl, xgb_clf.pkl, lap_rf.pkl, laps_rf.pkl, best_model.pkl

import os, re, warnings, joblib
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import top_k_accuracy_score
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- CONFIG ----------
CUTOFF_YEAR = 2015        # set to 2015 or 2018 to exclude retired drivers
MIN_SAMPLES_PER_DRIVER = 2
TEST_SIZE = 0.25
RANDOM_STATE = 42
# --------------------------

def read_csv_safe(path):
    if not os.path.exists(path):
        return None
    # read as strings first to avoid \N unicodeescape issues
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])

def to_int_safe(x):
    try:
        if x is None: return None
        s = str(x).strip()
        if s in ("", "None", "nan", "NaN", r"\N"): return None
        m = re.search(r"-?\d+", s.replace(",",""))
        return int(m.group(0)) if m else None
    except:
        return None

print("[STEP] Loading CSV files from:", DATA_DIR)
races = read_csv_safe(os.path.join(DATA_DIR, "races.csv"))
results = read_csv_safe(os.path.join(DATA_DIR, "results.csv"))
lap_times = read_csv_safe(os.path.join(DATA_DIR, "lap_times.csv"))
drivers = read_csv_safe(os.path.join(DATA_DIR, "drivers.csv"))
constructors = read_csv_safe(os.path.join(DATA_DIR, "constructors.csv"))

if races is None or results is None:
    raise SystemExit("races.csv and results.csv are required in data/")

# Ensure no python unicodeescape failure: we treat string raw content
# Normalize column names to lowercase for robust lookup (keep originals but add lowercase mapping)
races.columns = [c.strip() for c in races.columns]
results.columns = [c.strip() for c in results.columns]
if lap_times is not None:
    lap_times.columns = [c.strip() for c in lap_times.columns]

# Parse numeric columns in results safely
for c in ["raceId","driverId","constructorId","grid","position","milliseconds","laps","points"]:
    if c in results.columns:
        results[c] = pd.to_numeric(results[c].replace(r"\N", ""), errors="coerce")

# Filter races by cutoff year if available
if "year" in races.columns:
    races["year"] = pd.to_numeric(races["year"].replace(r"\N",""), errors="coerce")
    race_keep_ids = set(races[races["year"].fillna(0).astype(int) >= CUTOFF_YEAR]["raceId"].astype(int).tolist())
    print(f"[INFO] Keeping races >= {CUTOFF_YEAR}: {len(race_keep_ids)}")
else:
    race_keep_ids = set(pd.to_numeric(races["raceId"], errors="coerce").dropna().astype(int).tolist())
    print("[WARN] races.csv missing 'year' column — keeping all races")

# filter results rows
results = results[results["raceId"].isin(race_keep_ids)].copy()
results = results[results["driverId"].notna()]
print("[INFO] Results rows after filtering:", len(results))

# Build race winners mapping (position == 1)
if "position" not in results.columns:
    raise SystemExit("results.csv must contain 'position' column")

winners = results[results["position"] == 1].copy()
winners["raceId"] = winners["raceId"].astype(int)
winners["driverId"] = winners["driverId"].astype(int)
race_winner = dict(zip(winners["raceId"].tolist(), winners["driverId"].tolist()))

# Keep only races that have a winner and at least 2 participants
race_counts = results.groupby("raceId")["driverId"].nunique()
valid_races = [rid for rid,c in race_counts.items() if c >= 2 and rid in race_winner]
results = results[results["raceId"].isin(valid_races)].copy()
print(f"[INFO] Valid races with winners & >=2 participants: {len(valid_races)}")

# Compute driver historical aggregates
results["position_num"] = pd.to_numeric(results["position"], errors="coerce")
driver_mean_finish = results.groupby("driverId")["position_num"].apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()).to_dict()

# latest constructor per driver
latest_constructor = results.sort_values("raceId").groupby("driverId")["constructorId"].last().dropna().astype(int).to_dict()

# lap_times aggregation if present
mean_lap_by_driver = {}
if lap_times is not None and "milliseconds" in lap_times.columns and "driverId" in lap_times.columns:
    lap_times["milliseconds"] = pd.to_numeric(lap_times["milliseconds"].replace(r"\N",""), errors="coerce")
    lap_times["driverId"] = pd.to_numeric(lap_times["driverId"].replace(r"\N",""), errors="coerce")
    mean_lap_by_driver = lap_times.groupby("driverId")["milliseconds"].mean().to_dict()

# Build dataset: one row per (race, driver) with label = race winner driverId
rows = []
for _, r in results.iterrows():
    rid = int(r["raceId"])
    did = int(r["driverId"])
    cid = int(r["constructorId"]) if not pd.isna(r.get("constructorId")) else 0
    if rid not in race_winner: continue
    label_winner = race_winner[rid]
    grid = int(r["grid"]) if not pd.isna(r.get("grid")) else 99
    dmean = float(driver_mean_finish.get(did, 99.0))
    cid_latest = int(latest_constructor.get(did, cid))
    avg_lap_ms = mean_lap_by_driver.get(did, np.nan)
    laps = int(r["laps"]) if not pd.isna(r.get("laps")) else np.nan

    # find circuit id from races table if present
    circuit_id = 0
    if "circuitId" in races.columns:
        try:
            # raceId may be string in races table; convert
            cand = races[races["raceId"].astype(int) == rid]
            if not cand.empty:
                circuit_id = int(cand.iloc[0].get("circuitId") or 0)
        except Exception:
            circuit_id = 0

    rows.append({
        "raceId": rid,
        "driverId": did,
        "constructorId": cid_latest,
        "grid": grid,
        "driver_mean_prev_finish": dmean,
        "circuitId": circuit_id,
        "label_winner": int(label_winner),
        "avg_lap_ms": float(avg_lap_ms) if not pd.isna(avg_lap_ms) else np.nan,
        "laps": laps
    })

df = pd.DataFrame(rows)
print("[INFO] Base training rows:", len(df), "unique drivers:", df['driverId'].nunique())

# remove drivers with very few samples (classes with < MIN_SAMPLES_PER_DRIVER)
driver_counts = df["driverId"].value_counts().to_dict()
valid_drivers = [int(d) for d,c in driver_counts.items() if c >= MIN_SAMPLES_PER_DRIVER]
df = df[df["driverId"].isin(valid_drivers)].copy()
print("[INFO] After removing rare drivers:", len(df), "drivers:", len(valid_drivers))

if df.empty:
    raise SystemExit("No training samples remain after filtering. Reduce CUTOFF_YEAR or MIN_SAMPLES_PER_DRIVER.")

# Prepare encoders
le_driver = LabelEncoder()
le_constructor = LabelEncoder()
le_circuit = LabelEncoder()

driver_ids_sorted = sorted([str(int(x)) for x in df["driverId"].unique().tolist()])
constructor_ids_sorted = sorted([str(int(x)) for x in df["constructorId"].unique().tolist()])
circuit_ids_sorted = sorted([str(int(x)) for x in df["circuitId"].unique().tolist()])

le_driver.fit(driver_ids_sorted)
le_constructor.fit(constructor_ids_sorted)
le_circuit.fit(circuit_ids_sorted)

# Target encoder: winner driverIds (strings)
target_classes = sorted(list(set([str(int(x)) for x in df["label_winner"].unique().tolist()])))
le_target = LabelEncoder(); le_target.fit(target_classes)

# Feature construction
df["constructor_str"] = df["constructorId"].astype(int).astype(str)
df["constructor_enc"] = le_constructor.transform(df["constructor_str"])
df["circuit_str"] = df["circuitId"].astype(int).astype(str)
df["circuit_enc"] = le_circuit.transform(df["circuit_str"])

feature_cols = ["grid", "driver_mean_prev_finish", "constructor_enc", "circuit_enc"]
X = df[feature_cols].fillna(0).astype(float).values
y_str = df["label_winner"].astype(int).astype(str).values
y = le_target.transform(y_str)  # contiguous labels 0..n-1

# scale
scaler = StandardScaler(); Xs = scaler.fit_transform(X)

# ensure labels have >=2 samples (required for stratify)
label_counts = Counter(y)
labels_single = [lab for lab,cnt in label_counts.items() if cnt < 2]
if labels_single:
    mask = ~pd.Series(y).isin(labels_single)
    Xs = Xs[mask.values]
    y = y[mask.values]
    df = df.iloc[mask.values.nonzero()[0]] if hasattr(df, "iloc") else df

# split
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

print("[INFO] Train samples:", len(X_train), "Test samples:", len(X_test), "Classes:", len(le_target.classes_))

# Train KNN
print("[TRAIN] KNN")
knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
knn.fit(X_train, y_train)
knn_cal = CalibratedClassifierCV(knn, method='sigmoid', cv=3)
knn_cal.fit(X_train, y_train)

# Train RF
print("[TRAIN] RandomForest")
rf = RandomForestClassifier(n_estimators=300, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rf_cal = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
rf_cal.fit(X_train, y_train)

# Train XGBoost
print("[TRAIN] XGBoost")
num_class = len(le_target.classes_)
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=num_class,
                            n_estimators=200, max_depth=6, learning_rate=0.1,
                            use_label_encoder=False, eval_metric='mlogloss',
                            verbosity=0, n_jobs=-1, random_state=RANDOM_STATE)
xgb_clf.fit(X_train, y_train)

# Evaluate (top-1, top-3)
def eval_model(name, clf):
    try:
        probs = clf.predict_proba(X_test)
    except Exception:
        preds = clf.predict(X_test)
        probs = np.zeros((len(preds), num_class))
        for i,p in enumerate(preds): probs[i,int(p)] = 1.0
    top1 = top_k_accuracy_score(y_test, probs, k=1)
    top3 = top_k_accuracy_score(y_test, probs, k=3)
    print(f"[EVAL] {name} top1: {top1:.4f}, top3: {top3:.4f}")
    return top1, top3

knn_top1, _ = eval_model("KNN (calibrated)", knn_cal)
rf_top1, _ = eval_model("RF (calibrated)", rf_cal)
xgb_top1, _ = eval_model("XGBoost", xgb_clf)

# Best model pick
scores = {"knn": knn_top1, "rf": rf_top1, "xgb": xgb_top1}
best_key = max(scores, key=scores.get)
best_model = {"knn": knn_cal, "rf": rf_cal, "xgb": xgb_clf}[best_key]
print("[RESULT] Best model:", best_key, "score:", scores[best_key])

# Train regressors for average lap seconds and number of laps if enough rows
df_reg = df.dropna(subset=["avg_lap_ms","laps"])
lap_rf = None; laps_rf = None
if len(df_reg) > 20:
    print("[TRAIN] regressors")
    X_reg = df_reg[feature_cols].fillna(0).astype(float).values
    X_reg_s = scaler.transform(X_reg)
    y_lap_s = (df_reg["avg_lap_ms"].astype(float).values) / 1000.0
    y_laps = df_reg["laps"].astype(float).values
    lap_rf = RandomForestRegressor(n_estimators=200, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1)
    laps_rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    lap_rf.fit(X_reg_s, y_lap_s)
    laps_rf.fit(X_reg_s, y_laps)
    joblib.dump(lap_rf, os.path.join(MODELS_DIR, "lap_rf.pkl"))
    joblib.dump(laps_rf, os.path.join(MODELS_DIR, "laps_rf.pkl"))
    print("[SAVED] regressors saved")

# Save everything
encoders = {
    "le_driver": le_driver,
    "le_constructor": le_constructor,
    "le_circuit": le_circuit,
    "le_target": le_target,
    "feature_cols": feature_cols
}
joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(knn_cal, os.path.join(MODELS_DIR, "knn_clf.pkl"))
joblib.dump(rf_cal, os.path.join(MODELS_DIR, "rf_clf.pkl"))
joblib.dump(xgb_clf, os.path.join(MODELS_DIR, "xgb_clf.pkl"))
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
joblib.dump({"note":"labels_mapped_with_le_target"}, os.path.join(MODELS_DIR, "xgb_remap_info.pkl"))

print("[DONE] models saved to", MODELS_DIR)
