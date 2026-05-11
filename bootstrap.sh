#!/bin/bash
set -e

echo "==> Connecting kubectl to EKS cluster..."
aws eks update-kubeconfig --name ml-pipeline --region us-east-1

echo "==> Installing NVIDIA device plugin..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

echo "==> Annotating service account with IRSA role..."
INFERENCE_ROLE_ARN=$(cd terraform && terraform output -raw inference_role_arn)
kubectl annotate serviceaccount default \
  eks.amazonaws.com/role-arn=$INFERENCE_ROLE_ARN \
  --overwrite

echo "==> Installing Cluster Autoscaler..."
helm repo add autoscaler https://kubernetes.github.io/autoscaler --force-update
helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
  --set autoDiscovery.clusterName=ml-pipeline \
  --set awsRegion=us-east-1

echo "==> Deploying platform with Helm..."
helm upgrade --install ml-pipeline ./helm/ml-pipeline \
  -f helm/ml-pipeline/values.yaml \
  -f helm/ml-pipeline/values.secret.yaml

echo "==> Triggering initial training job..."
kubectl create job retrain-init --from=cronjob/retrain-cronjob 2>/dev/null || echo "Job already exists, skipping"

echo ""
echo "Done. Watch pods with: kubectl get pods -w"
echo "Attach to training: kubectl attach \$(kubectl get pods | grep retrain-init | awk '{print \$1}')"
