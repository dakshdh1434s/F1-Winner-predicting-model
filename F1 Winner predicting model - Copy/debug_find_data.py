# debug_build_samples.py
import re, os, pandas as pd, numpy as np
DATA = "data"
RES_PATH = os.path.join(DATA, "results.csv")

def safe_read_all(path):
    # read everything as string to avoid pandas coercing '\N' -> NaN
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    except Exception as e:
        try:
            return pd.read_excel(path, dtype=str)
        except Exception as e2:
            raise RuntimeError(f"Cannot read {path}: {e} / {e2}")

def extract_int_from_str(s):
    if s is None: return None
    if isinstance(s, (int, float)) and not np.isnan(s):
        try:
            return int(float(s))
        except:
            return None
    s = str(s).strip()
    if s in ["", "\\N", "N", "None", "nan", "NaN"]:
        return None
    # find first integer in string
    m = re.search(r"(-?\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None

print("Reading results.csv from:", RES_PATH)
df = safe_read_all(RES_PATH)
print("Total rows read:", len(df))
print("Columns:", list(df.columns))
# print first 5 rows (string-safe)
print("\nFirst 10 rows (raw):")
print(df.head(10).to_string(index=False))

# check core columns presence
candidates = {"raceId": ["raceId","race_id","raceid"],
              "driverId": ["driverId","driver_id","driverid"],
              "constructorId": ["constructorId","constructor_id","constructorid"],
              "position": ["position","pos","positionText","positionOrder"],
              "grid": ["grid","gridPosition","gridposition"]}

mapped = {}
for key, hints in candidates.items():
    for h in hints:
        for c in df.columns:
            if c.lower() == h.lower() or h.lower() in c.lower():
                mapped[key] = c
                break
        if key in mapped: break
    if key not in mapped:
        mapped[key] = None

print("\nDetected column mapping:", mapped)

# show stats for these columns
for k, col in mapped.items():
    if col is None:
        print(f"Column for {k} not found in results.csv")
    else:
        vals = df[col].astype(str)
        unique_sample = vals.drop_duplicates().head(20).tolist()
        print(f"\nColumn '{col}' sample (first 20 unique): {unique_sample[:20]}")
        print(f"Non-empty count: {(vals.astype(str).str.strip() != '').sum()}, empty-like count: {(vals.astype(str).str.strip() == '').sum()}")

# Try to clean and count how many rows would survive
count_ok = 0
count_no_driver = 0
count_no_constructor = 0
count_no_race = 0
count_problem = 0
examples_problem = []

for i, row in df.iterrows():
    # get raw values (string)
    rid_raw = row.get(mapped["raceId"]) if mapped["raceId"] else None
    did_raw = row.get(mapped["driverId"]) if mapped["driverId"] else None
    cid_raw = row.get(mapped["constructorId"]) if mapped["constructorId"] else None
    pos_raw = row.get(mapped["position"]) if mapped["position"] else None
    grid_raw = row.get(mapped["grid"]) if mapped["grid"] else None

    rid = extract_int_from_str(rid_raw)
    did = extract_int_from_str(did_raw)
    cid = extract_int_from_str(cid_raw)
    pos = None
    if pos_raw is None:
        pos = None
    else:
        pos = extract_int_from_str(pos_raw)
        if pos is None and str(pos_raw).strip() in ["\\N","N",""]:
            # DNF sentinel will be handled later
            pos = None

    if did is None:
        count_no_driver += 1
        if len(examples_problem) < 10:
            examples_problem.append(("no_driver", i, rid_raw, did_raw, cid_raw, pos_raw, grid_raw))
        continue
    if cid is None:
        count_no_constructor += 1
        if len(examples_problem) < 10:
            examples_problem.append(("no_constructor", i, rid_raw, did_raw, cid_raw, pos_raw, grid_raw))
        continue
    if rid is None:
        count_no_race += 1
        if len(examples_problem) < 10:
            examples_problem.append(("no_race", i, rid_raw, did_raw, cid_raw, pos_raw, grid_raw))
        continue

    # row OK
    count_ok += 1

print("\nCleaning summary:")
print(" Rows total:", len(df))
print(" Rows OK (driver,constructor,race parsed):", count_ok)
print(" Rows missing driverId:", count_no_driver)
print(" Rows missing constructorId:", count_no_constructor)
print(" Rows missing raceId:", count_no_race)
print(" Examples of first problems (up to 10):")
for ex in examples_problem:
    print("  ", ex)

# Show how many positions are '\\N' or non-numeric
if mapped["position"]:
    pos_series = df[mapped["position"]].astype(str)
    non_numeric = pos_series[~pos_series.str.match(r"^\s*-?\d+\s*$", na=False)]
    print("\nposition non-numeric sample (first 20):", non_numeric.head(20).tolist())
    print("position non-numeric count:", len(non_numeric))

print("\nDiagnostic finished.")
