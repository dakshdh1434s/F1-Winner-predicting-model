# infer_predictions.py
import os, joblib, json, traceback
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")

# Settings you can tweak
ACTIVE_YEAR_MIN = 2018   # only show drivers who appeared in >= this year (reduce retired drivers)
TOP_N = 20               # top N drivers to return
FALLBACK_LAP = 95.0      # seconds fallback for lap time
FALLBACK_LAPS = 50       # fallback number of laps

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        # print("safe_load error", path, e)
        return None

# --- Load models/artifacts (cached) ---
_encoders = safe_load(os.path.join(MODELS_DIR, "encoders.pkl"))
scaler = safe_load(os.path.join(MODELS_DIR, "scaler.pkl"))
knn_clf = safe_load(os.path.join(MODELS_DIR, "knn_clf.pkl"))
rf_clf = safe_load(os.path.join(MODELS_DIR, "rf_clf.pkl"))
xgb_clf = safe_load(os.path.join(MODELS_DIR, "xgb_clf.pkl"))
best_model = safe_load(os.path.join(MODELS_DIR, "best_model.pkl"))
lap_rf = safe_load(os.path.join(MODELS_DIR, "lap_rf.pkl"))
laps_rf = safe_load(os.path.join(MODELS_DIR, "laps_rf.pkl"))

# Encoders structure (from train script)
# encoders = { "le_driver":..., "le_constructor":..., "le_circuit":..., "le_target":..., "feature_cols": [...] }
if _encoders is None:
    encoders = {}
    feature_cols = ["grid", "driver_mean_prev_finish", "constructor_enc", "circuit_enc"]
else:
    encoders = _encoders
    feature_cols = encoders.get("feature_cols", ["grid", "driver_mean_prev_finish", "constructor_enc", "circuit_enc"])
le_target = encoders.get("le_target", None)
le_constructor = encoders.get("le_constructor", None)
le_circuit = encoders.get("le_circuit", None)
le_driver = encoders.get("le_driver", None)

# --- CSV loading helpers ---
def read_csv(path):
    if not os.path.exists(path):
        return None
    # keep strings for safety and treat "\N" values as NA
    return pd.read_csv(path, dtype=str, na_values=["\\N", ""])

def prepare_dataframes():
    results = read_csv(os.path.join(DATA_DIR, "results.csv"))
    races = read_csv(os.path.join(DATA_DIR, "races.csv"))
    drivers = read_csv(os.path.join(DATA_DIR, "drivers.csv"))
    constructors = read_csv(os.path.join(DATA_DIR, "constructors.csv"))
    lap_times = read_csv(os.path.join(DATA_DIR, "lap_times.csv"))
    # convert numeric fields where needed
    if results is not None:
        for c in ["raceId","driverId","constructorId","grid","position","milliseconds","laps"]:
            if c in results.columns:
                results[c] = pd.to_numeric(results[c], errors="coerce")
    if races is not None and "year" in races.columns:
        races["year"] = pd.to_numeric(races["year"], errors="coerce")
    if lap_times is not None and "milliseconds" in lap_times.columns:
        lap_times["milliseconds"] = pd.to_numeric(lap_times["milliseconds"], errors="coerce")
    return results, races, drivers, constructors, lap_times

# Cached dataframes
_results_df = None
_races_df = None
_drivers_df = None
_constructors_df = None
_lap_times_df = None

def load_all():
    global _results_df, _races_df, _drivers_df, _constructors_df, _lap_times_df
    if _results_df is None:
        _results_df, _races_df, _drivers_df, _constructors_df, _lap_times_df = prepare_dataframes()
    return _results_df, _races_df, _drivers_df, _constructors_df, _lap_times_df

# find raceId by name (exact or fuzzy)
def find_race_id_by_name(name):
    results = load_all()[0]
    races = load_all()[1]
    if races is None:
        return None
    if not name:
        return None
    name_l = name.strip().lower()
    # try exact match on 'name' column
    if "name" in races.columns:
        cand = races[races["name"].str.lower() == name_l]
        if not cand.empty:
            return int(cand.iloc[-1]["raceId"])
        cand = races[races["name"].str.lower().str.contains(name_l, na=False)]
        if not cand.empty:
            return int(cand.iloc[-1]["raceId"])
        # token search
        for token in name_l.split():
            cand = races[races["name"].str.lower().str.contains(token, na=False)]
            if not cand.empty:
                return int(cand.iloc[-1]["raceId"])
    return None

# build per-driver historical stats up to (but not including) target race
def build_driver_stats_before(race_id):
    results, races, drivers, constructors, lap_times = load_all()
    if results is None:
        return pd.DataFrame()
    # if races table has dates, use date ordering; else fallback numeric raceId
    try:
        if races is not None and "raceId" in races.columns and "date" in races.columns:
            # find date of race_id
            cand = races[races["raceId"].astype(float) == float(race_id)]
            if not cand.empty:
                rdate = pd.to_datetime(cand.iloc[-1]["date"], errors="coerce")
                prev_race_ids = races[pd.to_datetime(races["date"], errors="coerce") < rdate]["raceId"].dropna().astype(int).tolist()
                prev_results = results[results["raceId"].astype(float).isin(prev_race_ids)]
            else:
                prev_results = results[results["raceId"].astype(float) < float(race_id)]
        else:
            prev_results = results[results["raceId"].astype(float) < float(race_id)]
    except Exception:
        prev_results = results[results["raceId"].astype(float) < float(race_id)]
    if prev_results is None or prev_results.shape[0] == 0:
        return pd.DataFrame()

    g = prev_results.groupby("driverId").agg(
        mean_finish = ("position", "mean"),
        mean_lap_ms = ("milliseconds","mean"),
        mean_grid = ("grid","mean"),
        last_year = ("raceId", "max"),
        appearances = ("position","count")
    ).reset_index()
    # last_year here is raceId; we can map to year via races dataframe if present
    if races is not None and "raceId" in races.columns and "year" in races.columns:
        race_to_year = races.set_index(races["raceId"].astype(float))["year"].to_dict()
        def last_year_val(rid):
            try:
                return int(race_to_year.get(float(rid), np.nan))
            except:
                return np.nan
        g["last_year"] = g["last_year"].apply(last_year_val)
    else:
        g["last_year"] = np.nan
    # convert driverId numeric
    g["driverId"] = g["driverId"].astype(float).astype(int)
    # compute mean_lap_sec
    g["mean_lap_sec"] = g["mean_lap_ms"] / 1000.0
    g = g.fillna({"mean_finish": 99.0, "mean_lap_sec": float(FALLBACK_LAP), "mean_grid": 99.0, "appearances": 0})
    return g[["driverId","mean_finish","mean_lap_sec","mean_grid","appearances","last_year"]]

# driver & team strings
def build_name_maps():
    _, _, drivers, constructors, _ = load_all()
    dmap = {}
    cmap = {}
    if drivers is not None:
        for _, r in drivers.iterrows():
            try:
                did = int(float(r.get("driverId")))
            except:
                continue
            first = r.get("forename") or r.get("givenName") or r.get("name") or r.get("firstName") or ""
            last = r.get("surname") or r.get("familyName") or r.get("lastName") or ""
            fullname = (str(first) + " " + str(last)).strip()
            if not fullname:
                fullname = r.get("driverRef") or r.get("name") or f"Driver {did}"
            dmap[did] = fullname
    if constructors is not None:
        for _, r in constructors.iterrows():
            try:
                cid = int(float(r.get("constructorId")))
            except:
                continue
            cname = r.get("name") or r.get("constructorRef") or f"Team {cid}"
            cmap[cid] = cname
    return dmap, cmap

# mean lap per driver from lap_times.csv (fallback/regressor training)
def mean_lap_times():
    _, _, _, _, lap_times = load_all()
    if lap_times is None or lap_times.shape[0]==0:
        return {}
    if "milliseconds" in lap_times.columns and "driverId" in lap_times.columns:
        lap_times["milliseconds"] = pd.to_numeric(lap_times["milliseconds"], errors="coerce")
        lap_times["driverId"] = pd.to_numeric(lap_times["driverId"], errors="coerce")
        g = lap_times.groupby("driverId")["milliseconds"].mean()
        return {int(d): float(ms)/1000.0 for d, ms in g.to_dict().items() if not pd.isna(ms)}
    return {}

# latest constructor per driver using results
def latest_constructor_for_drivers():
    results, _, _, _, _ = load_all()
    if results is None: return {}
    try:
        # sort by raceId ascending so .last() gives latest constructor
        df = results.sort_values("raceId")
        last = df.groupby("driverId")["constructorId"].last()
        return {int(k): int(v) for k,v in last.dropna().to_dict().items()}
    except Exception:
        return {}

# main inference function
def infer(grand_prix_name="", date_str="", weather="", model_choice="best", top_n=TOP_N, active_year_min=ACTIVE_YEAR_MIN):
    """
    Returns list of dict rows:
      { rank, driver (full name), team, prob (0..1), pred_lap (sec), pred_laps (int) }
    """
    try:
        # load data
        results, races, drivers, constructors, lap_times = load_all()
        name_map, team_map = build_name_maps()
        mean_lap_map = mean_lap_times()
        latest_cons = latest_constructor_for_drivers()

        # find raceId
        race_id = find_race_id_by_name(grand_prix_name)
        # if not found, choose latest race available
        if race_id is None:
            if races is not None and "raceId" in races.columns:
                try:
                    race_id = int(races["raceId"].astype(float).max())
                except:
                    race_id = None

        if results is None or race_id is None:
            return []

        participants = results[results["raceId"].astype(float) == float(race_id)].copy()
        # if no participants found, try sprint_results or fallback to last race participants
        if participants.shape[0] == 0:
            # fallback last race participants
            last_rid = results["raceId"].astype(float).max()
            participants = results[results["raceId"].astype(float) == last_rid].copy()
        if participants.shape[0] == 0:
            return []

        # only unique driverIds
        participants = participants[participants["driverId"].notna()]
        participants["driverId"] = participants["driverId"].astype(int)
        unique_drivers = sorted(participants["driverId"].unique().tolist())

        # build driver stats before this race
        stats = build_driver_stats_before(race_id)
        # merge participants with stats
        dfp = pd.DataFrame({"driverId": unique_drivers})
        dfp = dfp.merge(stats, on="driverId", how="left")
        dfp["mean_finish"] = dfp["mean_finish"].fillna(99.0)
        dfp["mean_lap_sec"] = dfp["mean_lap_sec"].fillna(FALLBACK_LAP)
        dfp["mean_grid"] = dfp["mean_grid"].fillna(99.0)
        dfp["appearances"] = dfp["appearances"].fillna(0)
        dfp["last_year"] = dfp["last_year"].fillna(np.nan)

        # filter out drivers who haven't appeared since ACTIVE_YEAR_MIN if requested (reduces retired drivers)
        if active_year_min and "last_year" in dfp.columns:
            # if last_year is NaN we keep them (no info)
            def active_filter(row):
                ly = row.get("last_year")
                try:
                    if pd.isna(ly): return True
                    return int(ly) >= int(active_year_min)
                except:
                    return True
            dfp["is_active"] = dfp.apply(active_filter, axis=1)
            dfp = dfp[dfp["is_active"] == True].copy()
            if dfp.shape[0] == 0:
                # if filter removes all, fallback to no filter
                dfp = pd.DataFrame({"driverId": unique_drivers}).merge(stats, on="driverId", how="left")
                dfp["mean_finish"] = dfp["mean_finish"].fillna(99.0)
                dfp["mean_lap_sec"] = dfp["mean_lap_sec"].fillna(FALLBACK_LAP)
                dfp["mean_grid"] = dfp["mean_grid"].fillna(99.0)
                dfp["appearances"] = dfp["appearances"].fillna(0)

        # build features according to training feature_cols if available
        # encode constructor and circuit using encoders if possible using latest constructor for driver and circuit of race
        constructor_ids = []
        for did in dfp["driverId"].tolist():
            constructor_ids.append(latest_cons.get(int(did), np.nan))
        dfp["constructorId"] = constructor_ids

        # circuit id from races
        circuit_id = None
        races_df = races
        if races_df is not None and "raceId" in races_df.columns and "circuitId" in races_df.columns:
            cand = races_df[races_df["raceId"].astype(float) == float(race_id)]
            if not cand.empty:
                try:
                    circuit_id = int(float(cand.iloc[-1]["circuitId"]))
                except:
                    circuit_id = None

        # encode constructor and circuit
        if le_constructor is not None:
            cons_str = dfp["constructorId"].fillna(0).astype(int).astype(str).tolist()
            # ensure unseen encodings handled by mapping to nearest or 0 index; we will fallback to 0 for unseen
            try:
                cons_enc = []
                for s in cons_str:
                    if s in le_constructor.classes_:
                        cons_enc.append(int(np.where(le_constructor.classes_ == s)[0][0]))
                    else:
                        # fallback to 0 index
                        cons_enc.append(0)
                dfp["constructor_enc"] = cons_enc
            except Exception:
                dfp["constructor_enc"] = 0
        else:
            dfp["constructor_enc"] = 0

        if le_circuit is not None and circuit_id is not None:
            s = str(int(circuit_id))
            if s in le_circuit.classes_:
                dfp["circuit_enc"] = int(np.where(le_circuit.classes_ == s)[0][0])
            else:
                dfp["circuit_enc"] = 0
        else:
            dfp["circuit_enc"] = 0

        # final feature matrix
        # training used ["grid","driver_mean_prev_finish","constructor_enc","circuit_enc"]
        Xfeat = pd.DataFrame()
        if "grid" in feature_cols:
            # grid is not always available for prediction; use mean_grid or 99
            Xfeat["grid"] = dfp.get("mean_grid", dfp.get("appearances", 99)).fillna(99.0)
        # driver_mean_prev_finish -> mean_finish
        Xfeat["driver_mean_prev_finish"] = dfp.get("mean_finish", 99.0).astype(float)
        # constructor_enc, circuit_enc
        Xfeat["constructor_enc"] = dfp.get("constructor_enc", 0).astype(float)
        Xfeat["circuit_enc"] = dfp.get("circuit_enc", 0).astype(float)

        # scale
        try:
            if scaler is not None:
                Xs = scaler.transform(Xfeat.values)
            else:
                Xs = Xfeat.values
        except Exception:
            Xs = Xfeat.values

        # pick model
        def choose_model(choice):
            # choice: 'best','xgb','rf','knn'
            if choice == "xgb" and xgb_clf is not None:
                return xgb_clf, "XGBoost"
            if choice == "rf" and rf_clf is not None:
                return rf_clf, "RandomForest"
            if choice == "knn" and knn_clf is not None:
                return knn_clf, "KNN"
            if choice == "best" and best_model is not None:
                return best_model, "Best"
            # fallback order
            for m,name in [(best_model,"Best"),(xgb_clf,"XGBoost"),(rf_clf,"RandomForest"),(knn_clf,"KNN")]:
                if m is not None:
                    return m,name
            return None, "None"

        model_obj, model_name = choose_model(model_choice)

        # Obtain probabilities for each driver: need to locate, for each driver, the class index in le_target
        probs = None
        if model_obj is None or le_target is None:
            # fallback small uniform probabilities
            probs = np.ones(len(dfp)) / len(dfp)
        else:
            # model.predict_proba(Xs) returns shape (n_samples, n_classes)
            try:
                raw = model_obj.predict_proba(Xs)
                # get mapping from class_index -> driverId (strings in le_target.classes_)
                # le_target.classes_ are strings of driverIds (as saved in train script)
                classes = [int(x) for x in le_target.classes_.tolist()]
                cls_to_index = {classes[i]: i for i in range(len(classes))}
                probs_list = []
                for i, did in enumerate(dfp["driverId"].tolist()):
                    cls_idx = cls_to_index.get(int(did), None)
                    if cls_idx is None:
                        probs_list.append(1e-6)
                    else:
                        # raw row length should equal len(classes); but if classes mismatch, guard
                        try:
                            probs_list.append(float(raw[i, cls_idx]))
                        except Exception:
                            # try using predicted probability of predicted class
                            prow = raw[i]
                            probs_list.append(float(np.max(prow)))
                probs = np.array(probs_list)
            except Exception:
                # sometimes calibrated classifiers have different behaviour; fallback to predicted probabilities by one-vs-rest or use predict
                try:
                    preds = model_obj.predict(Xs)
                    # if preds are encoded target indices, use simple mapping: predicted driver gets 1.0 prob
                    probs = np.zeros(len(dfp))
                    for i,p in enumerate(preds):
                        try:
                            # if p is int label index into le_target.classes_
                            probs[i] = 1.0
                        except:
                            probs[i] = 0.0
                except Exception:
                    probs = np.ones(len(dfp)) / len(dfp)

        # generate predicted lap (seconds) and predicted laps using regressors if available else mean or fallback
        pred_laps_arr = []
        pred_lap_arr = []
        # prepare regressor inputs same scaling
        try:
            if lap_rf is not None and laps_rf is not None:
                predicted_lap = lap_rf.predict(Xs)
                predicted_laps = laps_rf.predict(Xs)
                pred_lap_arr = [float(max(20.0, p)) for p in predicted_lap]  # ensure plausible
                pred_laps_arr = [int(max(1, round(p))) for p in predicted_laps]
            else:
                # regressors missing -> use mean_lap_map or fallback
                for did in dfp["driverId"].tolist():
                    pl = mean_lap_map.get(int(did), FALLBACK_LAP)
                    pred_lap_arr.append(float(pl))
                    pred_laps_arr.append(int(FALLBACK_LAPS))
        except Exception:
            # fallback conservative
            for did in dfp["driverId"].tolist():
                pl = mean_lap_map.get(int(did), FALLBACK_LAP)
                pred_lap_arr.append(float(pl))
                pred_laps_arr.append(int(FALLBACK_LAPS))

        # build rows and return top_n by prob
        dfp["prob"] = probs
        dfp["pred_lap"] = pred_lap_arr
        dfp["pred_laps"] = pred_laps_arr

        # ensure driver names and team names
        def driver_name(did):
            return name_map.get(int(did), f"Driver {int(did)}")
        def team_name(did):
            cid = latest_cons.get(int(did), np.nan)
            return team_map.get(int(cid), f"Team {int(cid)}") if not pd.isna(cid) else "Unknown"

        dfp["driver_name"] = dfp["driverId"].apply(driver_name)
        dfp["team_name"] = dfp["driverId"].apply(team_name)

        dfp_sorted = dfp.sort_values("prob", ascending=False).head(top_n).reset_index(drop=True)
        rows = []
        for i, r in dfp_sorted.iterrows():
            rows.append({
                "rank": int(i+1),
                "driver": r["driver_name"],
                "team": r["team_name"],
                "prob": float(r["prob"]),
                "pred_lap": float(r["pred_lap"]),
                "pred_laps": int(r["pred_laps"])
            })
        return rows, model_name
    except Exception as e:
        traceback.print_exc()
        return [], "error"
