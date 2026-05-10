terraform {
    backend "s3" {
        bucket = "ml-pipeline-models-tanish"
        key    = "terraform/state"
        region = "us-east-1"
    }

    required_providers {
        aws = {
            source = "hashicorp/aws"
            version = "~> 5.0"
        }
        tls = {
            source = "hashicorp/tls"
            version = "~> 4.0"
        }
    }

    required_version = ">= 1.0"
}

provider "aws" {
    region = var.aws_region
}