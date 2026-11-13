# vertex-ai-ml-pipeline

A training pipeline for Vertex AI Pipelines (Kubeflow Pipelines v2) with a quality gate: data is validated before training, the model is evaluated on held-out rows, and it is registered in the Vertex AI Model Registry only if it clears the bar. Infrastructure is defined in Terraform.

## Pipeline

```
ingest --> validate --> train --> evaluate --> [quality gate: AUC >= min_auc on >= 30 held-out rows]
             |                                        |
       fails the run on                         register in Vertex AI Model Registry
       bad data                                 (new version under the same model name)
```

| Step | What it does | Stops the run when |
|---|---|---|
| ingest | Loads a labelled table (bundled demo data, local CSV, or `gs://` CSV) | the source can't be read |
| validate | Checks schema, row count, nulls, binary target, class balance | any check fails |
| train | Stratified split, median imputation fitted on training rows only, scikit-learn model (`logreg` or `gbdt`), fixed seed | |
| evaluate | AUC, accuracy, F1 on the held-out split; logs metrics | |
| register | Uploads the model as a new registry version | never runs unless the gate passes |

Why it's built this way:

- **Logic is separate from orchestration.** Everything in `mlpipe/steps.py` is plain Python with unit tests. The KFP components in `mlpipe/components.py` are thin wrappers.
- **One pinned image for every step.** Steps run in an image you build and push, with nothing installed at run time, so a run is reproducible and doesn't depend on PyPI being reachable.
- **The gate is part of the graph**, not a convention. The compiled spec contains `register` only inside the condition, and a test asserts that.
- **Least privilege.** Terraform creates a dedicated service account with Vertex AI user, read access to the image repository, and write access to one artifact bucket.

## Results on the demo dataset

Scikit-learn's bundled breast cancer dataset (569 rows, 30 features), 20% held out (114 rows), seed 42:

| Model | AUC | Accuracy | F1 |
|---|---|---|---|
| logreg | 0.9954 | 0.9825 | 0.9861 |
| gbdt | 0.9907 | 0.9561 | 0.9660 |

These come from running `mlpipe.steps` locally with scikit-learn 1.9. It is a small, easy dataset used to exercise the pipeline, and a single split, so treat the numbers as a smoke test and not as a benchmark. A test also trains on shuffled labels and confirms that model fails the gate.

## Run it

Local, no cloud needed:

```bash
pip install ".[pipeline]" pytest pyyaml
pytest -q                                   # 19 tests
python -m mlpipe.pipeline --out pipeline.yaml
```

On Google Cloud:

```bash
# 1. Infrastructure
terraform -chdir=terraform init
terraform -chdir=terraform apply -var project_id=$PROJECT -var region=$REGION

# 2. Step image
IMAGE=$REGION-docker.pkg.dev/$PROJECT/ml-pipelines/mlpipe:$(git rev-parse --short HEAD)
gcloud builds submit --tag $IMAGE .

# 3. Compile against that image and submit
PIPELINE_IMAGE=$IMAGE python -m mlpipe.pipeline --out pipeline.yaml
python -m mlpipe.submit --project $PROJECT --region $REGION \
  --bucket $PROJECT-ml-pipeline-artifacts \
  --service-account ml-pipeline@$PROJECT.iam.gserviceaccount.com \
  --serving-image <Vertex AI prebuilt scikit-learn prediction image>
```

`--serving-image` is required and has no default on purpose: pick the Vertex AI prebuilt scikit-learn prediction container whose scikit-learn version matches the one in your step image, from the current list in the Vertex AI documentation. A mismatch can fail when the model is loaded for serving.

## Tests

- `tests/test_steps.py`: validation rejects seven kinds of bad data; training is reproducible for a fixed seed; the held-out split is stratified; missing values are imputed from training medians; a model trained on shuffled labels fails the gate.
- `tests/test_pipeline.py`: the pipeline compiles; training depends on validation; `register` exists only inside the gate; no step installs packages at run time; the component bodies run end to end with stand-in artifacts.

CI runs the tests, compiles the pipeline and uploads the spec, and runs `terraform validate`.

## Status and limits

- Tested locally on Python 3.13 with kfp 2.18 and scikit-learn 1.9: the step logic, the component bodies and pipeline compilation. The pipeline has not been submitted to Vertex AI from this repository, and the Terraform has not been applied or validated outside CI. Expect to adjust details on the first real run.
- The saved model is the scikit-learn estimator. Imputation medians are stored beside it (`medians.json`) but a prebuilt serving container won't apply them, so send complete rows at prediction time or use a custom prediction routine.
- The gate is an absolute AUC threshold on one split. Planned: compare against the current registry champion, add cross-validation, data drift checks against the last training set, and an optional deploy step with a canary.
- Binary classification on numeric features only.

## License

MIT
