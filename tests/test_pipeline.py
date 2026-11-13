"""Compile the pipeline and run the component bodies locally with stand-in artifacts."""
import yaml

from mlpipe import components
from mlpipe.pipeline import compile_pipeline


class Artifact:
    def __init__(self, path):
        self.path, self.uri, self.metadata, self.metrics = str(path), f"gs://bucket/{path.name}", {}, {}

    def log_metric(self, key, value):
        self.metrics[key] = value


def test_pipeline_compiles_with_gate(tmp_path):
    spec = yaml.safe_load(open(compile_pipeline(str(tmp_path / "pipeline.yaml"))))
    tasks = spec["root"]["dag"]["tasks"]
    assert {"ingest-op", "validate-op", "train-op", "evaluate-op"} <= set(tasks)
    assert "validate-op" in tasks["train-op"]["dependentTasks"]        # training waits for validation
    gate = next(t for name, t in tasks.items() if "quality-gate" in name or "condition" in name)
    assert "evaluate-op" in gate["dependentTasks"] and "triggerPolicy" in gate
    gated = spec["components"][gate["componentRef"]["name"]]["dag"]["tasks"]
    assert set(gated) == {"register-op"}                              # register only inside the gate
    params = spec["root"]["inputDefinitions"]["parameters"]
    assert params["min_auc"]["defaultValue"] == 0.95
    assert not params["serving_image"].get("isOptional", False)       # must be supplied
    images = {e["container"]["image"] for e in spec["deploymentSpec"]["executors"].values()}
    assert images == {components.IMAGE}
    commands = [" ".join(e["container"]["command"]) for e in spec["deploymentSpec"]["executors"].values()]
    assert not any("pip install" in c for c in commands)              # nothing is installed at run time


def test_component_bodies_run_end_to_end(tmp_path):
    dataset, report = Artifact(tmp_path / "data.csv"), Artifact(tmp_path / "report")
    model, test_data, metrics = Artifact(tmp_path / "model"), Artifact(tmp_path / "test.csv"), Artifact(tmp_path / "m")

    assert components.ingest_op.python_func("builtin://breast_cancer", dataset) == 569
    components.validate_op.python_func(dataset, 200, report)
    assert report.metrics["rows"] == 569
    components.train_op.python_func(dataset, "logreg", 42, model, test_data)
    assert model.metadata["framework"] == "scikit-learn"
    assert components.evaluate_op.python_func(model, test_data, 0.95, metrics) is True
    assert metrics.metrics["gate_passed"] == 1 and metrics.metrics["auc"] > 0.95
    assert components.evaluate_op.python_func(model, test_data, 0.9999, Artifact(tmp_path / "m2")) is False
