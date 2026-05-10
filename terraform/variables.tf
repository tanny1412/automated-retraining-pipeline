variable "aws_region" {
    description = "AWS region to deploy resources"
    type        = string
    default     = "us-east-1"
  }

variable "cluster_name" {
    description = "EKS cluster name"
    type        = string
    default     = "ml-pipeline"
  }
  
variable "s3_bucket" {
    description = "S3 bucket for model weights"
    type        = string
    default     = "ml-pipeline-models"
  }

 variable "ecr_inference_repo" {
    description = "ECR repository name for inference image"
    type        = string
    default     = "ml-pipeline-inference"
  }
  
  variable "ecr_training_repo" {
    description = "ECR repository name for training image"
    type        = string
    default     = "ml-pipeline-training"
  }