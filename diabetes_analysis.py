import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw charts to PNG files, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns


# ===========================================================================
# SECTION: CONFIG
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_PATH = DATA_DIR / "diabetes.csv"

ID_COLUMN = "PatientID"
TARGET_COLUMN = "Outcome"

# TUPLE: the feature names never change, so an immutable tuple is the right fit.
FEATURE_COLUMNS = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)

# LIST: columns where a value of 0 is physiologically impossible,
# so 0 really means "not recorded" (missing value).
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# DICTIONARY: allowed (min, max) range for every column. Used when adding/updating records.
# For ZERO_AS_MISSING columns, 0 is accepted and means "unknown".
VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0, 70),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (1, 120),
    "Outcome": (0, 1),
}

# Columns that must be whole numbers
INTEGER_COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "Age", "Outcome"]


# ===========================================================================
# SECTION: GENERATE DATASET
# ===========================================================================
N_RECORDS = 768
SEED = 42
TARGET_POSITIVE_RATE = 0.349


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _calibrate_intercept(base_score, target_rate):
    """Bisection search so the average predicted probability hits the target rate."""
    low, high = -30.0, 10.0
    for _ in range(80):
        mid = (low + high) / 2
        if _sigmoid(base_score + mid).mean() > target_rate:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def build_dataset(n=N_RECORDS, seed=SEED):
    rng = np.random.default_rng(seed)

    age = np.clip(np.round(21 + rng.gamma(2.0, 6.0, n)), 21, 81).astype(int)
    bmi = np.clip(rng.normal(32.0, 6.8, n), 18.2, 67.1)
    pregnancies = np.clip(rng.poisson(np.maximum((age - 18) / 4.0, 0.3)), 0, 17)
    glucose = np.clip(rng.normal(116 + 0.35 * (age - 33) + 0.9 * (bmi - 32), 27), 44, 199)
    blood_pressure = np.clip(rng.normal(69 + 0.3 * (age - 33) + 0.3 * (bmi - 32), 11), 24, 122)
    skin = np.clip(rng.normal(20 + 0.6 * (bmi - 32), 8), 7, 99)
    insulin = np.clip(rng.lognormal(4.55 + 0.012 * (glucose - 121), 0.6), 14, 846)
    dpf = np.clip(rng.gamma(2.0, 0.24, n) + 0.08, 0.078, 2.42)

    base_score = 0.035 * glucose + 0.075 * bmi + 0.028 * age + 0.9 * dpf + 0.04 * pregnancies
    prob = _sigmoid(base_score + _calibrate_intercept(base_score, TARGET_POSITIVE_RATE))
    outcome = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            ID_COLUMN: np.arange(1, n + 1),
            "Pregnancies": pregnancies,
            "Glucose": np.round(glucose).astype(int),
            "BloodPressure": np.round(blood_pressure).astype(int),
            "SkinThickness": np.round(skin).astype(int),
            "Insulin": np.round(insulin).astype(int),
            "BMI": np.round(bmi, 1),
            "DiabetesPedigreeFunction": np.round(dpf, 3),
            "Age": age,
            "Outcome": outcome,
        }
    )

    # Mimic the real dataset: missing measurements are stored as 0.
    missing_rates = {"Glucose": 0.007, "BloodPressure": 0.045, "SkinThickness": 0.296, "Insulin": 0.487, "BMI": 0.014}
    for col, rate in missing_rates.items():
        df.loc[rng.random(n) < rate, col] = 0
    return df


def create_dataset_file(force=False):
    """Write the CSV unless it already exists. Returns True if a file was written."""
    if DATASET_PATH.exists() and not force:
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_dataset().to_csv(DATASET_PATH, index=False)
    return True


# ===========================================================================
# SECTION: UTILS
# ===========================================================================
LINE_WIDTH = 78


# --------------------------------------------------------------------------- #
# Display helpers (user-defined functions)
# --------------------------------------------------------------------------- #
def print_header(title):
    """Print a centred section title between two lines."""
    print("\n" + "=" * LINE_WIDTH)
    print(title.center(LINE_WIDTH))
    print("=" * LINE_WIDTH)


def print_subheader(title):
    print(f"\n--- {title} " + "-" * max(0, LINE_WIDTH - len(title) - 5))


def print_table(df, index=False, max_rows=None):
    """Print a DataFrame as a neat text table."""
    if df is None or len(df) == 0:
        print("  (no rows to show)")
        return
    shown = df if max_rows is None else df.head(max_rows)
    print(shown.to_string(index=index))


def pause():
    input("\nPress Enter to continue...")


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def prompt_int(label, low=None, high=None):
    """Ask until the user types a valid whole number (within optional limits)."""
    while True:
        text = input(label).strip()
        try:
            value = int(text)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if (low is not None and value < low) or (high is not None and value > high):
            print(f"  Value must be between {low} and {high}.")
            continue
        return value


def prompt_field(column, current=None):
    """
    Ask for one column value, validating against VALID_RANGES.
    If `current` is given, pressing Enter keeps the current value (used when updating).
    """
    low, high = VALID_RANGES[column]
    hint = f"[{low}-{high}]"
    if column in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
        hint += " (0 = not recorded)"
    suffix = f" (Enter = keep {current})" if current is not None else ""
    while True:
        text = input(f"  {column} {hint}{suffix}: ").strip()
        if text == "" and current is not None:
            return current
        try:
            value = int(text) if column in INTEGER_COLUMNS else float(text)
        except ValueError:
            print("   Invalid number, try again.")
            continue
        if not low <= value <= high:
            print(f"   Must be between {low} and {high}.")
            continue
        return value


# --------------------------------------------------------------------------- #
# Python Data Structures demo (list, tuple, set, dict + list/dict methods)
# --------------------------------------------------------------------------- #
def data_structures_demo(df):
    """Show lists, tuples, sets, dictionaries, list methods and dict methods on real data."""
    print_header("PYTHON DATA STRUCTURES DEMO")

    # 1) LIST + 2) LIST METHODS ------------------------------------------------
    print_subheader("1 & 2. List and list methods (first 8 glucose readings)")
    glucose_list = df["Glucose"].head(8).round(0).astype(int).tolist()
    print("  original            :", glucose_list)
    glucose_list.append(150)
    print("  append(150)         :", glucose_list)
    glucose_list.extend([90, 90])
    print("  extend([90, 90])    :", glucose_list)
    glucose_list.insert(0, 100)
    print("  insert(0, 100)      :", glucose_list)
    removed = glucose_list.pop()
    print(f"  pop() removed {removed}  :", glucose_list)
    glucose_list.sort()
    print("  sort()              :", glucose_list)
    glucose_list.reverse()
    print("  reverse()           :", glucose_list)
    print("  count(90) / index(150) / len():",
          glucose_list.count(90), "/", glucose_list.index(150), "/", len(glucose_list))
    high_readings = [g for g in glucose_list if g >= 126]          # list comprehension
    print("  list comprehension (glucose >= 126):", high_readings)

    # 3) TUPLE ---------------------------------------------------------------
    print_subheader("3. Tuple (immutable) - the feature names")
    print("  FEATURE_COLUMNS      :", FEATURE_COLUMNS)
    print("  first item / slice   :", FEATURE_COLUMNS[0], "/", FEATURE_COLUMNS[1:3])
    age_range = (int(df["Age"].min()), int(df["Age"].max()))       # tuple packing
    youngest, oldest = age_range                                    # tuple unpacking
    print(f"  age_range tuple      : {age_range} -> youngest={youngest}, oldest={oldest}")
    try:
        FEATURE_COLUMNS[0] = "Changed"                              # tuples cannot be modified
    except TypeError as error:
        print("  tuple is immutable   :", error)

    # 4) SET -----------------------------------------------------------------
    print_subheader("4. Set - group patients by risk markers (unique IDs only)")
    high_glucose = set(df.loc[df["Glucose"] >= 140, ID_COLUMN])
    obese = set(df.loc[df["BMI"] >= 30, ID_COLUMN])
    print(f"  high glucose (>=140): {len(high_glucose)} patients | obese (BMI>=30): {len(obese)} patients")
    print("  intersection (both)       :", len(high_glucose & obese))
    print("  union (either)            :", len(high_glucose | obese))
    print("  difference (glucose only) :", len(high_glucose - obese))
    print("  symmetric difference      :", len(high_glucose ^ obese))
    print("  unique pregnancy counts   :", sorted(set(df["Pregnancies"].astype(int))))

    # 5) DICTIONARY + DICTIONARY METHODS ---------------------------------------
    print_subheader("5. Dictionary and dictionary methods")
    summary = {
        "patients": len(df),
        "avg_age": round(float(df["Age"].mean()), 1),
        "avg_glucose": round(float(df["Glucose"].mean()), 1),
        "positives": int(df[TARGET_COLUMN].sum()),
    }
    print("  summary dict          :", summary)
    print("  keys()                :", list(summary.keys()))
    print("  values()              :", list(summary.values()))
    print("  get('avg_bmi', 'n/a') :", summary.get("avg_bmi", "n/a"))
    summary.update({"avg_bmi": round(float(df["BMI"].mean()), 1)})
    print("  after update()        :", summary)
    summary.setdefault("negatives", summary["patients"] - summary["positives"])
    print("  after setdefault()    :", summary)
    print("  pop('negatives')      :", summary.pop("negatives"))
    for key, value in summary.items():                              # items()
        print(f"    {key:<12} -> {value}")
    squares = {col: len(col) for col in FEATURE_COLUMNS[:3]}       # dict comprehension
    print("  dict comprehension    :", squares)


# ===========================================================================
# SECTION: DATA HANDLER
# ===========================================================================
def load_dataset(path=DATASET_PATH):
    """Read the CSV into a DataFrame. Adds a PatientID column if the file has none."""
    df = pd.read_csv(path)
    if ID_COLUMN not in df.columns:
        df.insert(0, ID_COLUMN, range(1, len(df) + 1))
    return df


def save_dataset(df, path=DATASET_PATH):
    """Write the DataFrame back to CSV (called after every add / update / delete)."""
    df.to_csv(path, index=False)


def backup_dataset(path=DATASET_PATH):
    """Copy the CSV to <name>_backup.csv before risky changes."""
    backup_path = path.with_name(path.stem + "_backup.csv")
    shutil.copy(path, backup_path)
    return backup_path


# --------------------------------------------------------------------------- #
# File handling with plain Python (open / csv module)
# --------------------------------------------------------------------------- #
def read_raw_csv(path=DATASET_PATH):
    """Read the file line by line with the csv module. Returns (header, rows)."""
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


def file_summary(path=DATASET_PATH):
    """Process the raw file with plain Python and return a dictionary of facts about it."""
    header, rows = read_raw_csv(path)
    positives = 0
    outcome_index = header.index(TARGET_COLUMN)
    for row in rows:                       # manual processing without pandas
        if row[outcome_index] == "1":
            positives += 1
    return {
        "file": str(path),
        "size_kb": round(path.stat().st_size / 1024, 1),
        "columns": header,
        "records": len(rows),
        "positives_counted_manually": positives,
        "first_row": dict(zip(header, rows[0])) if rows else {},
    }


def export_text_report(text, path):
    """Write any text (e.g. the summary report) to a .txt file."""
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)
    return path


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def missing_value_report(df):
    """Count 'hidden' missing values (zeros in columns where 0 is impossible)."""
    rows = []
    for col in ZERO_AS_MISSING:
        count = int((df[col] == 0).sum())
        rows.append({"Column": col, "Missing (0)": count, "Percent": round(100 * count / len(df), 1)})
    return pd.DataFrame(rows)


def clean_dataset(df):
    """
    Return a cleaned COPY:
      1. zeros in ZERO_AS_MISSING columns -> NaN
      2. NaN -> median of that column (robust against outliers)
    The raw DataFrame is left untouched, so the CSV keeps what was recorded.
    """
    cleaned = df.copy()
    for col in ZERO_AS_MISSING:
        cleaned[col] = cleaned[col].replace(0, np.nan)
    for col in ZERO_AS_MISSING:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    return cleaned


def nan_count_table(df):
    """NaN counts per feature column (after replacing zeros) - shows the effect of cleaning."""
    temp = df.copy()
    for col in ZERO_AS_MISSING:
        temp[col] = temp[col].replace(0, np.nan)
    return temp[list(FEATURE_COLUMNS)].isna().sum()


# ===========================================================================
# SECTION: RECORD MANAGER
# ===========================================================================
ALL_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]


def validate_record(record):
    """Check a dict of values against VALID_RANGES. Returns a list of error messages."""
    errors = []
    for column, (low, high) in VALID_RANGES.items():
        if column not in record:
            errors.append(f"{column} is missing")
        elif not low <= record[column] <= high:
            errors.append(f"{column}={record[column]} is outside {low}-{high}")
    return errors


def next_patient_id(df):
    return int(df[ID_COLUMN].max()) + 1 if len(df) else 1


def add_record(df, record):
    """Append a new patient. Returns (new_df, new_id)."""
    errors = validate_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    new_id = next_patient_id(df)
    row = {ID_COLUMN: new_id, **record}
    new_df = pd.concat([df, pd.DataFrame([row])[ALL_COLUMNS]], ignore_index=True)
    for col in INTEGER_COLUMNS:
        new_df[col] = new_df[col].astype(int)
    return new_df, new_id


def display_records(df, page_size=20):
    """Show ALL records, one page at a time (Enter = next page, q = stop)."""
    total = len(df)
    for start in range(0, total, page_size):
        print_table(df.iloc[start:start + page_size])
        end = min(start + page_size, total)
        print(f"\nShowing {start + 1}-{end} of {total}")
        if end < total:
            if input("Enter = next page, q = stop: ").strip().lower() == "q":
                break


def find_by_id(df, patient_id):
    """Return the matching row(s) for one PatientID (empty DataFrame if none)."""
    return df[df[ID_COLUMN] == patient_id]


def search_records(df, column, operator, value):
    """
    Generic search, e.g. search_records(df, "Age", ">=", 50).
    Supported operators: ==, !=, >, >=, <, <=
    """
    if column not in df.columns:
        raise KeyError(f"Unknown column '{column}'")
    series = df[column]
    operations = {
        "==": series == value,
        "!=": series != value,
        ">": series > value,
        ">=": series >= value,
        "<": series < value,
        "<=": series <= value,
    }
    if operator not in operations:
        raise ValueError(f"Operator must be one of {list(operations)}")
    return df[operations[operator]]


def update_record(df, patient_id, updates):
    """Change fields of one patient. `updates` is a dict {column: new_value}."""
    mask = df[ID_COLUMN] == patient_id
    if not mask.any():
        return df, False
    merged = {**df.loc[mask].iloc[0].to_dict(), **updates}
    errors = validate_record({c: merged[c] for c in VALID_RANGES})
    if errors:
        raise ValueError("; ".join(errors))
    new_df = df.copy()
    for column, value in updates.items():
        new_df.loc[mask, column] = value
    for col in INTEGER_COLUMNS:
        new_df[col] = new_df[col].astype(int)
    return new_df, True


def delete_record(df, patient_id):
    """Remove one patient. Returns (new_df, deleted_flag)."""
    mask = df[ID_COLUMN] == patient_id
    if not mask.any():
        return df, False
    return df[~mask].reset_index(drop=True), True


def show_record_card(row):
    """Print one patient as a vertical card."""
    print_subheader(f"Patient #{int(row[ID_COLUMN])}")
    for column in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        value = int(row[column]) if column in INTEGER_COLUMNS else row[column]
        print(f"  {column:<26}: {value}")


# ===========================================================================
# SECTION: ANALYSIS
# ===========================================================================
# =========================================================================== #
# NumPy
# =========================================================================== #
def numpy_demo(df):
    """1D/2D arrays, indexing, slicing, sum/mean/max/min and standard deviation."""
    print_header("NUMPY DEMO")

    # --- Creating arrays ------------------------------------------------------
    print_subheader("Creating arrays")
    manual_1d = np.array([85, 110, 140, 165, 190])
    print("  manual 1D array          :", manual_1d, "| shape", manual_1d.shape)
    glucose = df["Glucose"].to_numpy()
    print(f"  1D array from dataset    : Glucose, shape {glucose.shape}, dtype {glucose.dtype}")
    matrix = df[["Glucose", "BMI", "Age"]].to_numpy()
    print(f"  2D array from dataset    : [Glucose, BMI, Age], shape {matrix.shape}")
    print("  np.zeros((2,3))          :\n", np.zeros((2, 3)))
    print("  np.arange(0, 100, 25)    :", np.arange(0, 100, 25))

    # --- Indexing and slicing -------------------------------------------------
    print_subheader("Indexing and slicing")
    print("  glucose[0] (first)       :", glucose[0])
    print("  glucose[-1] (last)       :", glucose[-1])
    print("  glucose[:5]              :", glucose[:5])
    print("  glucose[10:15]           :", glucose[10:15])
    print("  glucose[::200] (every 200th):", glucose[::200])
    print("  matrix[0, 1] (row0, BMI) :", matrix[0, 1])
    print("  matrix[:3, :2] (3 rows, Glucose+BMI):\n", matrix[:3, :2])
    print("  matrix[:5, 2] (Age, 5 rows):", matrix[:5, 2])
    high = matrix[matrix[:, 0] >= 180]
    print(f"  boolean mask Glucose>=180 -> {len(high)} rows, first 3:\n", high[:3])

    # --- Aggregations ---------------------------------------------------------
    print_subheader("Sum / mean / max / min / standard deviation")
    stats = {
        "sum": float(np.sum(glucose)),
        "mean": float(np.mean(glucose)),
        "max": float(np.max(glucose)),
        "min": float(np.min(glucose)),
        "std (population)": float(np.std(glucose)),
        "std (sample, ddof=1)": float(np.std(glucose, ddof=1)),
    }
    for name, value in stats.items():
        print(f"  Glucose {name:<22}: {value:,.2f}")
    print("\n  2D column-wise (axis=0) for [Glucose, BMI, Age]:")
    for label, values in (("mean", matrix.mean(axis=0)), ("max", matrix.max(axis=0)),
                          ("min", matrix.min(axis=0)), ("std", matrix.std(axis=0))):
        print(f"    {label:<5}", np.round(values, 2))
    return stats


# =========================================================================== #
# Pandas
# =========================================================================== #
def pandas_demo(raw, clean):
    """DataFrame creation, selection, filtering, sorting, GroupBy, missing values, statistics."""
    print_header("PANDAS DEMO")

    print_subheader("Create a DataFrame (from a dictionary)")
    small = pd.DataFrame({"Name": ["A", "B", "C"], "Glucose": [95, 130, 170], "BMI": [22.5, 31.0, 38.2]})
    print_table(small)

    print_subheader("Read CSV -> shape and dtypes")
    print("  shape:", raw.shape)
    print(raw.dtypes.to_string())

    print_subheader("Selected rows and columns")
    print("  First 5 rows, columns Age/Glucose/BMI/Outcome  (.loc):")
    print_table(raw.loc[:4, ["Age", "Glucose", "BMI", "Outcome"]])
    print("\n  Rows 10-12, first 4 columns  (.iloc):")
    print_table(raw.iloc[10:13, :4])

    print_subheader("Filtering")
    older_high = clean[(clean["Age"] >= 50) & (clean["Glucose"] >= 140)]
    print(f"  Age >= 50 AND Glucose >= 140 -> {len(older_high)} patients")
    print_table(older_high[[ID_COLUMN, "Age", "Glucose", "BMI", "Outcome"]], max_rows=5)

    print_subheader("Sorting")
    print("  Top 5 by Glucose (descending):")
    print_table(clean.sort_values("Glucose", ascending=False)[[ID_COLUMN, "Glucose", "BMI", "Age", "Outcome"]].head(5))

    print_subheader("GroupBy")
    print("  Mean of key measurements per Outcome:")
    print_table(clean.groupby(TARGET_COLUMN)[["Age", "Glucose", "BMI", "BloodPressure"]].mean().round(2), index=True)

    print_subheader("Handling missing values")
    print("  Zeros recorded in the raw file (impossible values = missing):")
    print_table(missing_value_report(raw))
    print("  -> zeros replaced with NaN, then filled with the column median (see clean_dataset()).")
    print(f"  Zeros remaining after cleaning (Insulin): {(clean['Insulin'] == 0).sum()}")

    print_subheader("Statistical analysis")
    print_table(clean[list(FEATURE_COLUMNS)].describe().round(2), index=True)


# =========================================================================== #
# The five project questions
# =========================================================================== #
def basic_statistics(clean):
    """Answers to questions 1-3."""
    total = len(clean)
    positives = int((clean[TARGET_COLUMN] == 1).sum())
    return {
        "total_individuals": total,
        "average_age": round(float(clean["Age"].mean()), 2),
        "average_glucose": round(float(clean["Glucose"].mean()), 2),
        "positive_cases": positives,
        "negative_cases": total - positives,
        "positive_percent": round(100 * positives / total, 1),
    }


def describe_strength(r):
    """Translate a correlation value into words."""
    size = abs(r)
    if size < 0.1:
        return "negligible"
    if size < 0.3:
        return "weak"
    if size < 0.5:
        return "moderate"
    return "strong"


def bmi_outcome_association(clean):
    """Answer to question 4: group means, correlation and positive-rate by BMI category."""
    group_stats = clean.groupby(TARGET_COLUMN)["BMI"].agg(["count", "mean", "median", "std"]).round(2)
    r = float(np.corrcoef(clean["BMI"], clean[TARGET_COLUMN])[0, 1])
    labels = ["Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25-29.9)", "Obese I (30-34.9)", "Obese II+ (35+)"]
    bands = pd.cut(clean["BMI"], bins=[0, 18.5, 25, 30, 35, 200], labels=labels, right=False)
    by_band = clean.groupby(bands, observed=True)[TARGET_COLUMN].agg(patients="count", positives="sum", positive_rate="mean")
    by_band["positive_rate"] = (by_band["positive_rate"] * 100).round(1)
    return {"group_stats": group_stats, "correlation": round(r, 3), "strength": describe_strength(r), "by_band": by_band}


def correlations_with_outcome(clean):
    """Correlation of every feature with Outcome, strongest first."""
    corr = clean[list(FEATURE_COLUMNS)].corrwith(clean[TARGET_COLUMN]).round(3)
    return corr.sort_values(key=abs, ascending=False)


def rate_by_group(clean, column, bins, labels):
    """Positive rate (%) of Outcome for bins of one column (re-used for Age and Glucose)."""
    groups = pd.cut(clean[column], bins=bins, labels=labels, right=False)
    table = clean.groupby(groups, observed=True)[TARGET_COLUMN].agg(patients="count", positive_rate="mean")
    table["positive_rate"] = (table["positive_rate"] * 100).round(1)
    return table


def interesting_patterns(clean):
    """Answer to question 5: returns tables plus plain-English insights built from the data."""
    corr = correlations_with_outcome(clean)
    age_table = rate_by_group(clean, "Age", [0, 30, 40, 50, 60, 200], ["<30", "30-39", "40-49", "50-59", "60+"])
    glucose_table = rate_by_group(clean, "Glucose", [0, 100, 126, 160, 400],
                                  ["<100 (normal)", "100-125", "126-159", "160+"])
    preg_table = rate_by_group(clean, "Pregnancies", [0, 1, 4, 8, 50], ["0", "1-3", "4-7", "8+"])
    means = clean.groupby(TARGET_COLUMN)[list(FEATURE_COLUMNS)].mean()

    insights = []
    top_feature, top_value = corr.index[0], corr.iloc[0]
    insights.append(f"{top_feature} has the strongest link with Outcome (r = {top_value:+.2f}, {describe_strength(top_value)}).")
    insights.append(f"Average glucose is {means.loc[1, 'Glucose']:.0f} for positive vs {means.loc[0, 'Glucose']:.0f} for negative individuals "
                    f"(+{means.loc[1, 'Glucose'] - means.loc[0, 'Glucose']:.0f}).")
    insights.append(f"Positive rate rises from {glucose_table['positive_rate'].iloc[0]}% (glucose {glucose_table.index[0]}) "
                    f"to {glucose_table['positive_rate'].iloc[-1]}% (glucose {glucose_table.index[-1]}).")
    peak_band = age_table["positive_rate"].idxmax()
    insights.append(f"Positive rate is {age_table['positive_rate'].iloc[0]}% under 30 and peaks at "
                    f"{age_table['positive_rate'].max()}% in the {peak_band} band "
                    f"(only {int(age_table.loc[peak_band, 'patients'])} people, so treat with caution).")
    insights.append(f"Pregnancies and Age are themselves correlated (r = {clean['Pregnancies'].corr(clean['Age']):+.2f}), "
                    "so part of the Pregnancies effect may simply reflect age.")
    return {"correlations": corr, "age": age_table, "glucose": glucose_table,
            "pregnancies": preg_table, "means": means.round(2), "insights": insights}


# =========================================================================== #
# Terminal presentation
# =========================================================================== #
def print_core_answers(clean):
    """Print answers to the five project questions."""
    print_header("ANSWERS TO THE PROJECT QUESTIONS")
    stats = basic_statistics(clean)
    print(f"\n  Individuals analysed                     : {stats['total_individuals']}")
    print(f"  1. Average age                           : {stats['average_age']} years")
    print(f"  2. Average glucose level                 : {stats['average_glucose']}")
    print(f"  3. Positive diabetes outcomes            : {stats['positive_cases']} ({stats['positive_percent']}%)"
          f"  | negative: {stats['negative_cases']}")

    print_subheader("4. Is BMI associated with diabetes outcome?")
    bmi = bmi_outcome_association(clean)
    print("  BMI by outcome (0 = negative, 1 = positive):")
    print_table(bmi["group_stats"], index=True)
    print(f"\n  Correlation BMI <-> Outcome: r = {bmi['correlation']:+.3f}  ({bmi['strength']} positive association)")
    print("\n  Positive rate by BMI category:")
    print_table(bmi["by_band"], index=True)
    print("\n  Note: correlation shows association, not cause.")

    print_subheader("5. Which variables show interesting patterns?")
    patterns = interesting_patterns(clean)
    print("  Correlation of each variable with Outcome:")
    print_table(patterns["correlations"].to_frame("r"), index=True)
    print("\n  Positive rate (%) by age group:")
    print_table(patterns["age"], index=True)
    print("\n  Positive rate (%) by glucose band:")
    print_table(patterns["glucose"], index=True)
    print("\n  Key insights:")
    for number, text in enumerate(patterns["insights"], start=1):
        print(f"   {number}. {text}")


def build_summary_text(raw, clean):
    """Return the whole summary report as one string (so it can be printed AND saved to a file)."""
    stats = basic_statistics(clean)
    bmi = bmi_outcome_association(clean)
    patterns = interesting_patterns(clean)
    missing = missing_value_report(raw)

    lines = [
        "=" * 70,
        "DIABETES DATASET - SUMMARY REPORT (educational analysis, not a diagnosis)",
        "=" * 70,
        f"Records                : {stats['total_individuals']}",
        f"Average age            : {stats['average_age']} years",
        f"Average glucose        : {stats['average_glucose']}",
        f"Average BMI            : {clean['BMI'].mean():.2f}",
        f"Positive outcomes      : {stats['positive_cases']} ({stats['positive_percent']}%)",
        f"Negative outcomes      : {stats['negative_cases']}",
        "",
        "MISSING VALUES IN RAW FILE (zeros treated as missing, filled with median)",
        missing.to_string(index=False),
        "",
        f"BMI vs OUTCOME: r = {bmi['correlation']:+.3f} ({bmi['strength']})",
        bmi["by_band"].to_string(),
        "",
        "CORRELATION WITH OUTCOME",
        patterns["correlations"].to_string(),
        "",
        "AVERAGES BY OUTCOME",
        patterns["means"].T.to_string(),
        "",
        "KEY INSIGHTS",
        *[f"  {i}. {text}" for i, text in enumerate(patterns["insights"], start=1)],
        "",
        "Disclaimer: simulated/educational data; correlations are not causes.",
    ]
    return "\n".join(lines)


def summary_report(raw, clean):
    """Print the summary report and return its text."""
    text = build_summary_text(raw, clean)
    print_header("SUMMARY REPORT")
    print(text)
    return text


# ===========================================================================
# SECTION: VISUALIZATIONS
# ===========================================================================
sns.set_theme(style="whitegrid")
PALETTE = {0: "#4C9BE8", 1: "#E8744C"}


def _save(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_outcome_distribution(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette=PALETTE, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    ax.set(title="Diabetes outcome counts", xlabel="Outcome (0 = negative, 1 = positive)", ylabel="Individuals")
    return _save(fig, "01_outcome_distribution.png")


def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="Age", bins=20, kde=True, color="#4C9BE8", ax=ax)
    ax.axvline(df["Age"].mean(), color="red", linestyle="--", label=f"Mean {df['Age'].mean():.1f}")
    ax.legend()
    ax.set_title("Age distribution")
    return _save(fig, "02_age_distribution.png")


def plot_glucose_by_outcome(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="Glucose", hue=TARGET_COLUMN, palette=PALETTE, legend=False, ax=ax)
    ax.set_title("Glucose by diabetes outcome")
    return _save(fig, "03_glucose_by_outcome.png")


def plot_bmi_by_outcome(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="BMI", hue=TARGET_COLUMN, palette=PALETTE, legend=False, ax=axes[0])
    axes[0].set_title("BMI by outcome")
    band = bmi_outcome_association(df)["by_band"]
    sns.barplot(x=band.index.astype(str), y=band["positive_rate"], color="#E8744C", ax=axes[1])
    axes[1].set(title="Positive rate (%) by BMI category", xlabel="", ylabel="% positive")
    axes[1].tick_params(axis="x", rotation=25)
    return _save(fig, "04_bmi_vs_outcome.png")


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df[[*FEATURE_COLUMNS, TARGET_COLUMN]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    return _save(fig, "05_correlation_heatmap.png")


def plot_age_group_rate(df):
    table = rate_by_group(df, "Age", [0, 30, 40, 50, 60, 200], ["<30", "30-39", "40-49", "50-59", "60+"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=table.index.astype(str), y=table["positive_rate"], color="#4C9BE8", ax=ax)
    ax.set(title="Positive rate (%) by age group", xlabel="Age group", ylabel="% positive")
    return _save(fig, "06_positive_rate_by_age.png")


def generate_all_charts(df):
    """Create every chart and return the list of file paths."""
    makers = [plot_outcome_distribution, plot_age_distribution, plot_glucose_by_outcome,
              plot_bmi_by_outcome, plot_correlation_heatmap, plot_age_group_rate]
    return [maker(df) for maker in makers]


# ===========================================================================
# SECTION: MAIN
# ===========================================================================
MENU = """
  DATA & RECORDS                       ANALYSIS
  1. Dataset / file information         7. Answers to the 5 project questions
  2. Display all records                8. NumPy demo
  3. Add a new record                   9. Pandas demo
  4. Search for a record               10. Python data structures demo
  5. Update a record                   11. Summary report (+ save to file)
  6. Delete a record                   12. Generate charts (PNG files)
  0. Exit
"""


# --------------------------------------------------------------------------- #
# Menu actions
# --------------------------------------------------------------------------- #
def show_dataset_info(raw):
    print_header("DATASET / FILE INFORMATION")
    info = file_summary()
    print(f"  File                  : {info['file']}")
    print(f"  Size                  : {info['size_kb']} KB")
    print(f"  Records (plain Python): {info['records']}")
    print(f"  Positives (manual)    : {info['positives_counted_manually']}")
    print(f"  Columns               : {', '.join(info['columns'])}")
    print(f"  First row             : {info['first_row']}")
    print_subheader("Missing values (zeros) in the raw file")
    print_table(missing_value_report(raw))


def add_new_record(raw):
    print_header("ADD A NEW RECORD")
    record = {column: prompt_field(column) for column in [*FEATURE_COLUMNS, TARGET_COLUMN]}
    try:
        raw, new_id = add_record(raw, record)
    except ValueError as error:
        print("  Could not add record:", error)
        return raw
    save_dataset(raw)
    print(f"\n  Record added with PatientID {new_id}. Total records: {len(raw)}")
    return raw


def search_for_record(raw):
    print_header("SEARCH FOR A RECORD")
    print("  1. By PatientID\n  2. By condition (e.g. Age >= 50)")
    choice = input("  Choose: ").strip()
    if choice == "1":
        found = find_by_id(raw, prompt_int("  PatientID: ", 1))
        if found.empty:
            print("  No record found.")
        else:
            show_record_card(found.iloc[0])
    elif choice == "2":
        columns = list(raw.columns)
        column = input(f"  Column {columns[1:]}: ").strip()
        operator = input("  Operator (==, !=, >, >=, <, <=): ").strip()
        try:
            value = float(input("  Value: ").strip())
            results = search_records(raw, column, operator, value)
        except (ValueError, KeyError) as error:
            print("  Invalid search:", error)
            return
        print(f"\n  {len(results)} record(s) found.")
        print_table(results, max_rows=20)
        if len(results) > 20:
            print("  ... (showing first 20)")
    else:
        print("  Invalid choice.")


def update_existing_record(raw):
    print_header("UPDATE A RECORD")
    patient_id = prompt_int("  PatientID to update: ", 1)
    found = find_by_id(raw, patient_id)
    if found.empty:
        print("  No record found.")
        return raw
    row = found.iloc[0]
    show_record_card(row)
    print("\n  Enter new values (press Enter to keep the current value):")
    updates = {}
    for column in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        current = int(row[column]) if column in ("Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                                  "Insulin", "Age", "Outcome") else float(row[column])
        new_value = prompt_field(column, current=current)
        if new_value != current:
            updates[column] = new_value
    if not updates:
        print("  Nothing changed.")
        return raw
    raw, ok = update_record(raw, patient_id, updates)
    if ok:
        save_dataset(raw)
        print(f"  Updated {list(updates)} for patient #{patient_id}.")
    return raw


def delete_existing_record(raw):
    print_header("DELETE A RECORD")
    patient_id = prompt_int("  PatientID to delete: ", 1)
    found = find_by_id(raw, patient_id)
    if found.empty:
        print("  No record found.")
        return raw
    show_record_card(found.iloc[0])
    if input("\n  Delete this record permanently? (y/n): ").strip().lower() != "y":
        print("  Cancelled.")
        return raw
    backup_dataset()
    raw, ok = delete_record(raw, patient_id)
    if ok:
        save_dataset(raw)
        print(f"  Deleted patient #{patient_id}. A backup copy was saved. Total records: {len(raw)}")
    return raw


def make_report(raw, clean):
    text = summary_report(raw, clean)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = export_text_report(text, OUTPUT_DIR / "summary_report.txt")
    print(f"\n  Report saved to {path}")


def make_charts(clean):
    print_header("GENERATING CHARTS")
    for path in generate_all_charts(clean):
        print("  saved:", path)


# --------------------------------------------------------------------------- #
# Program flow
# --------------------------------------------------------------------------- #
def run_demo(raw):
    """Run everything once without any typing (handy for screenshots / marking)."""
    clean = clean_dataset(raw)
    show_dataset_info(raw)
    print_core_answers(clean)
    numpy_demo(clean)
    pandas_demo(raw, clean)
    data_structures_demo(clean)
    make_report(raw, clean)
    make_charts(clean)


def main():
    if create_dataset_file():
        print(f"Dataset not found - a simulated 768-record dataset was created at {DATASET_PATH}")
    raw = load_dataset()

    print_header("DIABETES DATASET ANALYSIS")
    print("  Educational analysis - NOT a medical diagnosis system.")
    print(f"  Loaded {len(raw)} records from {DATASET_PATH.name}")

    if "--demo" in sys.argv:
        run_demo(raw)
        return

    while True:
        print(MENU)
        choice = input("Choose an option (0-12): ").strip()
        clean = clean_dataset(raw)          # always analyse up-to-date data
        if choice == "1":
            show_dataset_info(raw)
        elif choice == "2":
            print_header("ALL RECORDS")
            display_records(raw)
        elif choice == "3":
            raw = add_new_record(raw)
        elif choice == "4":
            search_for_record(raw)
        elif choice == "5":
            raw = update_existing_record(raw)
        elif choice == "6":
            raw = delete_existing_record(raw)
        elif choice == "7":
            print_core_answers(clean)
        elif choice == "8":
            numpy_demo(clean)
        elif choice == "9":
            pandas_demo(raw, clean)
        elif choice == "10":
            data_structures_demo(clean)
        elif choice == "11":
            make_report(raw, clean)
        elif choice == "12":
            make_charts(clean)
        elif choice == "0":
            print("\nGoodbye!")
            break
        else:
            print("  Invalid option, please type a number from 0 to 12.")
            continue
        pause()


if __name__ == "__main__":
    main()
