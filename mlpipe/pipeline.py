"""Pipeline definition and compiler entry point.

  PIPELINE_IMAGE=... python -m mlpipe.pipeline --out pipeline.yaml"""
import argparse

from kfp import compiler, dsl

from .components import evaluate_op, ingest_op, register_op, train_op, validate_op

PIPELINE_NAME = "gated-training-pipeline"


@dsl.pipeline(name=PIPELINE_NAME,
              description="Ingest, validate, train, evaluate; register only if the quality gate passes.")
def training_pipeline(
    project: str,
    region: str,
    serving_image: str,
    source_uri: str = "builtin://breast_cancer",
    model_display_name: str = "gated-classifier",
    model_type: str = "logreg",
    min_rows: int = 200,
    min_auc: float = 0.95,
    seed: int = 42,
):
    data = ingest_op(source_uri=source_uri)
    checks = validate_op(dataset=data.outputs["dataset"], min_rows=min_rows)
    trained = train_op(dataset=data.outputs["dataset"], model_type=model_type, seed=seed).after(checks)
    evaluated = evaluate_op(model=trained.outputs["model"], test_data=trained.outputs["test_data"],
                            min_auc=min_auc)
    with dsl.If(evaluated.outputs["Output"] == True, name="quality-gate"):  # noqa: E712 (KFP needs ==)
        register_op(model=trained.outputs["model"], project=project, region=region,
                    display_name=model_display_name, serving_image=serving_image)


def compile_pipeline(path: str) -> str:
    compiler.Compiler().compile(training_pipeline, package_path=path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="pipeline.yaml")
    print("wrote", compile_pipeline(parser.parse_args().out))
