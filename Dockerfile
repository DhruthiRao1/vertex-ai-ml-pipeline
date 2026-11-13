# Image used by every pipeline step. Build and push to Artifact Registry, then
# compile the pipeline with PIPELINE_IMAGE set to the pushed tag.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml ./
COPY mlpipe ./mlpipe
RUN pip install ".[pipeline,gcs]"
RUN useradd --create-home app
USER app
