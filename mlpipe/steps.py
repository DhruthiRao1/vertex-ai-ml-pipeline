"""Pipeline logic as plain functions. The KFP components in components.py are
thin wrappers around these, so everything here is unit-testable without
Kubeflow or Google Cloud."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "target"
BUILTIN = "builtin://breast_cancer"
MODELS = ("logreg", "gbdt")


class DataValidationError(ValueError):
    pass


# --- ingest -----------------------------------------------------------------
def ingest(source_uri: str, out_csv: str) -> int:
    """Load a labelled table and write it as CSV. source_uri is the bundled demo
    dataset, a local CSV, or a gs:// CSV (needs gcsfs). Returns the row count."""
    if source_uri == BUILTIN:
        df = load_breast_cancer(as_frame=True).frame
        df.columns = [c.replace(" ", "_") for c in df.columns]
    else:
        df = pd.read_csv(source_uri)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return len(df)


# --- validate ---------------------------------------------------------------
@dataclass
class ValidationReport:
    rows: int
    features: int
    null_fraction: float
    minority_fraction: float
    duplicate_fraction: float


def validate(csv_path: str, min_rows: int = 200, max_null_fraction: float = 0.05,
             min_minority_fraction: float = 0.1) -> ValidationReport:
    """Fail fast on data that would train a misleading model."""
    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise DataValidationError(f"missing '{TARGET}' column")
    features = df.drop(columns=[TARGET])
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise DataValidationError(f"non-numeric feature columns: {non_numeric[:5]}")
    if len(df) < min_rows:
        raise DataValidationError(f"only {len(df)} rows, need at least {min_rows}")
    if df[TARGET].isna().any():
        raise DataValidationError("target has missing values")
    classes = set(df[TARGET].unique())
    if classes != {0, 1}:
        raise DataValidationError(f"target must be binary 0/1, found {sorted(classes)[:5]}")
    null_fraction = float(features.isna().to_numpy().mean()) if features.size else 0.0
    if null_fraction > max_null_fraction:
        raise DataValidationError(f"null fraction {null_fraction:.3f} exceeds {max_null_fraction}")
    minority = float(df[TARGET].value_counts(normalize=True).min())
    if minority < min_minority_fraction:
        raise DataValidationError(f"minority class is {minority:.3f} of rows, need {min_minority_fraction}")
    constant = [c for c in features.columns if features[c].nunique(dropna=True) <= 1]
    if len(constant) == features.shape[1]:
        raise DataValidationError("every feature is constant")
    return ValidationReport(len(df), features.shape[1], round(null_fraction, 5), round(minority, 5),
                            round(float(df.duplicated().mean()), 5))


# --- train ------------------------------------------------------------------
def _estimator(model_type: str, seed: int):
    if model_type == "logreg":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=seed))])
    if model_type == "gbdt":
        return Pipeline([("clf", GradientBoostingClassifier(random_state=seed))])
    raise ValueError(f"model_type must be one of {MODELS}")


def train(csv_path: str, model_dir: str, test_csv: str, model_type: str = "logreg",
          test_fraction: float = 0.2, seed: int = 42) -> dict:
    """Train on a stratified split. The held-out rows are written to test_csv and
    never seen during fitting. Median imputation is fitted on the training rows only."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(df, test_size=test_fraction, random_state=seed, stratify=df[TARGET])
    medians = train_df.drop(columns=[TARGET]).median()
    x_train = train_df.drop(columns=[TARGET]).fillna(medians)
    model = _estimator(model_type, seed).fit(x_train, train_df[TARGET])

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Vertex AI's prebuilt scikit-learn container expects the file to be named model.joblib.
    joblib.dump(model, out / "model.joblib")
    (out / "medians.json").write_text(json.dumps(medians.to_dict()))
    Path(test_csv).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_csv, index=False)
    return {"model_type": model_type, "train_rows": len(train_df), "test_rows": len(test_df),
            "features": list(x_train.columns), "seed": seed}


# --- evaluate ---------------------------------------------------------------
@dataclass
class EvalReport:
    auc: float
    accuracy: float
    f1: float
    test_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(model_dir: str, test_csv: str) -> EvalReport:
    model = joblib.load(Path(model_dir) / "model.joblib")
    medians = pd.Series(json.loads((Path(model_dir) / "medians.json").read_text()))
    test_df = pd.read_csv(test_csv)
    x = test_df.drop(columns=[TARGET]).fillna(medians)
    y = test_df[TARGET]
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return EvalReport(round(float(roc_auc_score(y, proba)), 5), round(float(accuracy_score(y, pred)), 5),
                      round(float(f1_score(y, pred)), 5), len(test_df))


def passes_gate(report: EvalReport, min_auc: float, min_test_rows: int = 30) -> bool:
    """A model is only registered if it clears the bar on enough held-out rows."""
    return report.test_rows >= min_test_rows and report.auc >= min_auc
