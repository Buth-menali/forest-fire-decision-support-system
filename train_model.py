import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# ----------------------------------------------------------
# 1. Load the 14 BehaviorSpace CSV exports
#    (skiprows=6 skips the metadata lines NetLogo adds above the header)
# ----------------------------------------------------------

CSV_FOLDER = "."   # folder that contains the CSV files

CSV_FILES = [
    "forestfire_forest_fire_experiment-table.csv",    # experiment 1
    "forestfire_forest_fire_experiment2-table.csv",
    "forestfire_forest_fire_experiment3-table.csv",
    "forestfire_forest_fire_experiment4-table.csv",
    "forestfire_forest_fire_experiment5-table.csv",
    "forestfire_forest_fire_experiment6-table.csv",
    "forestfire_forest_fire_experiment7-table.csv",
    "forestfire_forest_fire_experiment8-table.csv",
    "forestfire_forest_fire_experiment9-table.csv",
    "forestfire_forest_fire_experiment10-table.csv",
    "forestfire_forest_fire_experiment11-table.csv",
    "forestfire_forest_fire_experiment12-table.csv",
    "forestfire_forest_fire_experiment13-table.csv",
    "forestfire_forest_fire_experiment14-table.csv",
]

TIME_LIMIT = 500   # runs longer than this are treated as "did not finish"

frames = []

for number, name in enumerate(CSV_FILES, start=1):
    path = os.path.join(CSV_FOLDER, name)

    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping.")
        continue

    part = pd.read_csv(path, skiprows=6)
    part["source"] = f"exp{number}"
    frames.append(part)
    print(f"Loaded exp{number}: {len(part)} rows")

if not frames:
    raise SystemExit("No CSV files were loaded. Check CSV_FOLDER and CSV_FILES.")

df = pd.concat(frames, ignore_index=True)

# ----------------------------------------------------------
# 2. Keep only the LAST step of each run (the final outcome).
#    Run numbers repeat in every file, so group by source + run.
# ----------------------------------------------------------

last_step_idx = df.groupby(["source", "[run number]"])["[step]"].idxmax()
final_df = df.loc[last_step_idx].reset_index(drop=True)

print("\nTotal runs found:", len(final_df))

# ----------------------------------------------------------
# 3. Clean the data
# ----------------------------------------------------------

final_df["firebreak?"] = (
    final_df["firebreak?"].astype(str).str.lower().map({"true": 1, "false": 0})
)

features = [
    "density",
    "firebreak?",
    "regrowth-chance",
    "regrowth-time",
    "reignite-chance",
]
target = "percent-ever-burned"

final_df = final_df.dropna(subset=features + [target])

# Remove runs that never finished (fire kept re-igniting past the limit)
too_long = final_df["[step]"] > TIME_LIMIT
print(f"Removed {too_long.sum()} runs longer than {TIME_LIMIT} ticks")
final_df = final_df[~too_long].reset_index(drop=True)

print("Runs used for training:", len(final_df))
print(final_df["source"].value_counts().sort_index().to_string())

X = final_df[features]
y = final_df[target]

print("\nInput ranges in the data:")
print(X.describe().loc[["min", "max"]].round(2).to_string())

# ----------------------------------------------------------
# 4. Train the model
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 5. Check how good the model is (R2 and MAE)
# ----------------------------------------------------------

pred = model.predict(X_test)
mae_random = mean_absolute_error(y_test, pred)
r2_random = r2_score(y_test, pred)
print(f"\nRandom split    -> MAE: {mae_random:.2f} | R2: {r2_random:.3f}")

# Stricter test: whole setting combinations are held out, so the model
# is judged on settings it has never seen (repeated runs cannot leak).
groups = X.astype(str).agg("|".join, axis=1)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
g_train, g_test = next(gss.split(X, y, groups))

g_model = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
)
g_model.fit(X.iloc[g_train], y.iloc[g_train])
g_pred = g_model.predict(X.iloc[g_test])

mae_unseen = mean_absolute_error(y.iloc[g_test], g_pred)
r2_unseen = r2_score(y.iloc[g_test], g_pred)
print(f"Unseen settings -> MAE: {mae_unseen:.2f} | R2: {r2_unseen:.3f}")

print("\nFeature importance:")
importance = pd.Series(model.feature_importances_, index=features)
print(importance.sort_values(ascending=False).round(3).to_string())

# ----------------------------------------------------------
# 6. Save the final model (trained on ALL clean data)
# ----------------------------------------------------------

final_model = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
)
final_model.fit(X, y)

with open("fire_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

print("\nModel saved as fire_model.pkl")

# ----------------------------------------------------------
# 7. Save the scores so the Streamlit dashboard can show them
# ----------------------------------------------------------

metrics = {
    "r2": round(float(r2_random), 4),
    "mae": round(float(mae_random), 4),
    "r2_unseen": round(float(r2_unseen), 4),
    "mae_unseen": round(float(mae_unseen), 4),
    "training_runs": int(len(final_df)),
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Scores saved as model_metrics.json")