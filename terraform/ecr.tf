resource "aws_ecr_repository" "inference" {
  name         = var.ecr_inference_repo
  force_delete = true

  tags = {
    Project = var.cluster_name
  }
}

resource "aws_ecr_repository" "training" {
  name         = var.ecr_training_repo
  force_delete = true

  tags = {
    Project = var.cluster_name
  }
}
