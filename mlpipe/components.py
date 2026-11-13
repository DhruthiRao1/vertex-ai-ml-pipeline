"""KFP components. Each runs in the pipeline image (see Dockerfile), which has
this package installed, so the bodies only import and call mlpipe.steps.

Set PIPELINE_IMAGE before compiling, e.g.
  REGION-docker.pkg.dev/PROJECT/ml-pipelines/mlpipe:GIT_SHA"""
import os

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

IMAGE = os.environ.get("PIPELINE_IMAGE", "mlpipe:local")
# install_kfp_package=False: the image already contains kfp, so steps don't pip install at run time.


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def ingest_op(source_uri: str, dataset: Output[Dataset]) -> int:
    from mlpipe import steps
    rows = steps.ingest(source_uri, dataset.path)
    dataset.metadata["source_uri"] = source_uri
    dataset.metadata["rows"] = rows
    return rows


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def validate_op(dataset: Input[Dataset], min_rows: int, report: Output[Metrics]):
    from dataclasses import asdict

    from mlpipe import steps
    result = steps.validate(dataset.path, min_rows=min_rows)   # raises -> the pipeline stops here
    for key, value in asdict(result).items():
        report.log_metric(key, value)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def train_op(dataset: Input[Dataset], model_type: str, seed: int,
             model: Output[Model], test_data: Output[Dataset]):
    from mlpipe import steps
    info = steps.train(dataset.path, model.path, test_data.path, model_type=model_type, seed=seed)
    model.metadata["framework"] = "scikit-learn"
    model.metadata["model_type"] = info["model_type"]
    model.metadata["train_rows"] = info["train_rows"]
    model.metadata["seed"] = info["seed"]


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def evaluate_op(model: Input[Model], test_data: Input[Dataset], min_auc: float,
                metrics: Output[Metrics]) -> bool:
    from mlpipe import steps
    result = steps.evaluate(model.path, test_data.path)
    for key, value in result.to_dict().items():
        metrics.log_metric(key, value)
    passed = steps.passes_gate(result, min_auc)
    metrics.log_metric("gate_passed", int(passed))
    return passed


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def register_op(model: Input[Model], project: str, region: str, display_name: str,
                serving_image: str) -> str:
    """Upload to the Vertex AI Model Registry. Later runs add a new version under
    the same display name instead of creating a separate model."""
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    existing = aiplatform.Model.list(filter=f'display_name="{display_name}"')
    uploaded = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model.uri,
        serving_container_image_uri=serving_image,
        parent_model=existing[0].resource_name if existing else None,
        labels={"pipeline": "vertex-ai-ml-pipeline", "model_type": str(model.metadata.get("model_type", "unknown"))},
    )
    return uploaded.versioned_resource_name
