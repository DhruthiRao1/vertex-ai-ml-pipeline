"""Submit a compiled pipeline to Vertex AI Pipelines.

  python -m mlpipe.submit --project P --region asia-south1 --bucket my-bucket \\
      --service-account ml-pipeline@P.iam.gserviceaccount.com --serving-image IMAGE_URI"""
import argparse


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--bucket", required=True, help="GCS bucket name for pipeline artifacts")
    p.add_argument("--service-account", required=True)
    p.add_argument("--serving-image", required=True,
                   help="Vertex AI prebuilt scikit-learn prediction image matching the training sklearn version")
    p.add_argument("--template", default="pipeline.yaml")
    p.add_argument("--source-uri", default="builtin://breast_cancer")
    p.add_argument("--model-type", default="logreg", choices=["logreg", "gbdt"])
    p.add_argument("--min-auc", type=float, default=0.95)
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args(argv)

    from google.cloud import aiplatform
    aiplatform.init(project=a.project, location=a.region)
    job = aiplatform.PipelineJob(
        display_name="gated-training-pipeline",
        template_path=a.template,
        pipeline_root=f"gs://{a.bucket}/pipeline-root",
        parameter_values={"project": a.project, "region": a.region, "serving_image": a.serving_image,
                          "source_uri": a.source_uri, "model_type": a.model_type, "min_auc": a.min_auc},
        enable_caching=not a.no_cache,
    )
    job.submit(service_account=a.service_account)
    print("submitted:", job.resource_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
