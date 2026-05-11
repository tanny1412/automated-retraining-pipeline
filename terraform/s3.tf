  resource "aws_s3_bucket" "model_weights" {
    bucket        = var.s3_bucket
    force_destroy = true

    tags = {
      Name    = "ML Pipeline Model Weights"
      Project = var.cluster_name
    }
  }

  resource "aws_s3_bucket_versioning" "model_weights" {
    bucket = aws_s3_bucket.model_weights.id

    versioning_configuration {
      status = "Enabled" 
    }
  }