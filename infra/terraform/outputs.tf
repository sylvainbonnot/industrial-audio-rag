# Outputs for Industrial Audio RAG Infrastructure

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

# EKS Cluster Outputs
output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_version" {
  description = "The Kubernetes version for the cluster"
  value       = module.eks.cluster_version
}

# Node Group Outputs
output "node_groups" {
  description = "Map of attribute maps for all EKS managed node groups created"
  value       = module.eks.eks_managed_node_groups
  sensitive   = true
}

# IAM Role Outputs
output "cluster_autoscaler_role_arn" {
  description = "ARN of the cluster autoscaler IAM role"
  value       = module.cluster_autoscaler_irsa_role.iam_role_arn
}

output "load_balancer_controller_role_arn" {
  description = "ARN of the AWS Load Balancer Controller IAM role"
  value       = module.load_balancer_controller_irsa_role.iam_role_arn
}

output "ebs_csi_role_arn" {
  description = "ARN of the EBS CSI driver IAM role"
  value       = module.ebs_csi_irsa_role.iam_role_arn
}

# KMS Outputs
output "kms_key_arn" {
  description = "ARN of the KMS key used for EKS encryption"
  value       = aws_kms_key.eks.arn
}

output "kms_key_id" {
  description = "ID of the KMS key used for EKS encryption"
  value       = aws_kms_key.eks.key_id
}

# Application Outputs
output "application_url" {
  description = "URL to access the Industrial Audio RAG application"
  value = var.domain_name != "" ? (
    var.certificate_arn != "" ? "https://${var.domain_name}" : "http://${var.domain_name}"
  ) : "Use kubectl port-forward to access the application"
}

output "grafana_url" {
  description = "URL to access Grafana dashboard"
  value = var.enable_monitoring && var.domain_name != "" ? (
    var.certificate_arn != "" ? "https://grafana.${var.domain_name}" : "http://grafana.${var.domain_name}"
  ) : "Use kubectl port-forward to access Grafana"
}

# Cost Estimation Outputs
output "estimated_monthly_cost_usd" {
  description = "Estimated monthly cost in USD (approximate)"
  value = {
    eks_cluster = 72  # $0.10/hour * 24 * 30
    node_group_on_demand = var.node_group_desired_capacity * 30 * 24 * 0.0464  # t3.medium pricing
    ebs_storage = (var.node_group_desired_capacity * 100 + (var.enable_qdrant_persistence ? var.qdrant_storage_size : 0)) * 0.08  # gp3 pricing
    load_balancer = var.enable_ingress_controller ? 22.50 : 0  # NLB pricing
    total_estimated = 72 + (var.node_group_desired_capacity * 30 * 24 * 0.0464) + 
                     ((var.node_group_desired_capacity * 100 + (var.enable_qdrant_persistence ? var.qdrant_storage_size : 0)) * 0.08) +
                     (var.enable_ingress_controller ? 22.50 : 0)
  }
}

# Kubectl Configuration
output "kubectl_config" {
  description = "kubectl config command"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

# Helm Commands
output "helm_upgrade_command" {
  description = "Command to upgrade the application using Helm"
  value = "helm upgrade industrial-audio-rag ./deploy/helm --namespace default --set app.image.tag=NEW_TAG"
}

# Monitoring Access
output "monitoring_access" {
  description = "Commands to access monitoring tools"
  value = var.enable_monitoring ? {
    prometheus = "kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090"
    grafana    = "kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    grafana_password = "kubectl get secret -n monitoring prometheus-grafana -o jsonpath='{.data.admin-password}' | base64 --decode"
  } : null
}

# Security Information
output "security_groups" {
  description = "Security groups created"
  value = {
    eks_cluster_sg    = module.eks.cluster_security_group_id
    eks_additional_sg = aws_security_group.eks_additional.id
  }
}

# Backup Information
output "backup_configuration" {
  description = "Backup configuration details"
  value = {
    retention_days = var.backup_retention_days
    schedule      = var.backup_schedule
    ebs_snapshots = "Configured via AWS Backup (manual setup required)"
  }
}