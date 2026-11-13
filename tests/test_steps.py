import numpy as np
import pandas as pd
import pytest

from mlpipe import steps


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "data.csv"
    assert steps.ingest(steps.BUILTIN, str(path)) == 569
    return str(path)


def test_ingest_local_csv(tmp_path, data):
    out = tmp_path / "copy.csv"
    assert steps.ingest(data, str(out)) == 569


def test_validate_ok(data):
    report = steps.validate(data)
    assert report.rows == 569 and report.features == 30 and report.null_fraction == 0.0
    assert 0.3 < report.minority_fraction < 0.5


def _write(tmp_path, df):
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.drop(columns=["target"]), "missing 'target'"),
    (lambda d: d.head(50), "only 50 rows"),
    (lambda d: d.assign(target=d["target"] * 2), "binary 0/1"),
    (lambda d: d.assign(notes="text"), "non-numeric"),
    (lambda d: d.assign(mean_radius=np.where(d.index % 2 == 0, np.nan, d["mean_radius"]),
                        mean_texture=np.nan, mean_area=np.nan), "null fraction"),
    (lambda d: pd.concat([d[d.target == 0], d[d.target == 1].head(5)]), "minority class"),
    (lambda d: d.assign(target=np.where(d.index < 3, np.nan, d["target"])), "missing values"),
])
def test_validate_rejects_bad_data(tmp_path, data, mutate, message):
    with pytest.raises(steps.DataValidationError, match=message):
        steps.validate(_write(tmp_path, mutate(pd.read_csv(data))))


@pytest.mark.parametrize("model_type", steps.MODELS)
def test_train_evaluate_roundtrip(tmp_path, data, model_type):
    info = steps.train(data, str(tmp_path / "model"), str(tmp_path / "test.csv"), model_type=model_type)
    assert info["train_rows"] + info["test_rows"] == 569
    assert (tmp_path / "model" / "model.joblib").exists()
    report = steps.evaluate(str(tmp_path / "model"), str(tmp_path / "test.csv"))
    assert report.test_rows == info["test_rows"]
    assert report.auc > 0.95 and report.accuracy > 0.9


def test_training_is_reproducible(tmp_path, data):
    reports = []
    for run in ("a", "b"):
        steps.train(data, str(tmp_path / run), str(tmp_path / f"{run}.csv"), seed=7)
        reports.append(steps.evaluate(str(tmp_path / run), str(tmp_path / f"{run}.csv")))
    assert reports[0] == reports[1]


def test_held_out_rows_are_not_in_training_split(tmp_path, data):
    steps.train(data, str(tmp_path / "m"), str(tmp_path / "test.csv"), seed=1)
    full, test = pd.read_csv(data), pd.read_csv(tmp_path / "test.csv")
    assert len(test) == 114
    assert abs(test["target"].mean() - full["target"].mean()) < 0.01      # stratified


def test_missing_values_use_training_medians(tmp_path, data):
    df = pd.read_csv(data)
    df.loc[df.index % 25 == 0, "mean_radius"] = np.nan
    path = _write(tmp_path, df)
    steps.validate(path)
    steps.train(path, str(tmp_path / "m"), str(tmp_path / "test.csv"))
    assert steps.evaluate(str(tmp_path / "m"), str(tmp_path / "test.csv")).auc > 0.95


def test_shuffled_labels_fail_the_gate(tmp_path, data):
    """A model with no real signal must not be registered."""
    df = pd.read_csv(data)
    df["target"] = np.random.default_rng(0).permutation(df["target"].to_numpy())
    path = _write(tmp_path, df)
    steps.train(path, str(tmp_path / "m"), str(tmp_path / "test.csv"))
    report = steps.evaluate(str(tmp_path / "m"), str(tmp_path / "test.csv"))
    assert report.auc < 0.7
    assert not steps.passes_gate(report, min_auc=0.95)


def test_gate_rules():
    good = steps.EvalReport(auc=0.99, accuracy=0.97, f1=0.97, test_rows=114)
    assert steps.passes_gate(good, 0.95)
    assert not steps.passes_gate(good, 0.995)
    assert not steps.passes_gate(steps.EvalReport(1.0, 1.0, 1.0, test_rows=5), 0.95)


def test_bad_arguments(tmp_path, data):
    with pytest.raises(ValueError):
        steps.train(data, str(tmp_path / "m"), str(tmp_path / "t.csv"), model_type="xgboost")
    with pytest.raises(ValueError):
        steps.train(data, str(tmp_path / "m"), str(tmp_path / "t.csv"), test_fraction=0.9)
