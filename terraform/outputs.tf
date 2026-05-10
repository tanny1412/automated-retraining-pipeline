output "cluster_name" {
  value = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  value = aws_eks_cluster.main.endpoint
}

output "ecr_inference_url" {
  value = aws_ecr_repository.inference.repository_url
}

output "ecr_training_url" {
  value = aws_ecr_repository.training.repository_url
}
