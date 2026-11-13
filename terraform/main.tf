terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "asia-south1"
}

locals {
  apis = ["aiplatform.googleapis.com", "artifactregistry.googleapis.com", "storage.googleapis.com"]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

# Pipeline artifacts: datasets, models, metrics.
resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-ml-pipeline-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}

# Step images.
resource "google_artifact_registry_repository" "images" {
  repository_id = "ml-pipelines"
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# The identity pipeline steps run as. It gets only what the pipeline needs.
resource "google_service_account" "pipeline" {
  account_id   = "ml-pipeline"
  display_name = "Vertex AI training pipeline"
}

resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.pipeline.email}"
}

resource "google_storage_bucket_iam_member" "artifacts_rw" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.pipeline.email}"
}

resource "google_artifact_registry_repository_iam_member" "image_reader" {
  repository = google_artifact_registry_repository.images.name
  location   = var.region
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.pipeline.email}"
}

output "bucket" {
  value = google_storage_bucket.artifacts.name
}

output "service_account" {
  value = google_service_account.pipeline.email
}

output "image_repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
