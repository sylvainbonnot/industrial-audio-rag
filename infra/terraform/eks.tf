# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = local.cluster_name
  cluster_version = var.cluster_version
  
  # Networking
  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets
  
  # Enable public access with restricted CIDR blocks
  cluster_endpoint_public_access       = true
  cluster_endpoint_private_access      = true
  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks
  
  # Encryption
  cluster_encryption_config = var.enable_encryption ? [
    {
      provider_key_arn = aws_kms_key.eks.arn
      resources        = ["secrets"]
    }
  ] : []
  
  # Logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  cloudwatch_log_group_retention_in_days = var.log_retention_days
  
  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  # Node groups
  eks_managed_node_groups = {
    main = {
      name           = "${local.cluster_name}-main"
      instance_types = var.node_group_instance_types
      
      min_size     = var.node_group_min_capacity
      max_size     = var.node_group_max_capacity
      desired_size = var.node_group_desired_capacity
      
      # Use on-demand instances by default
      capacity_type = var.enable_spot_instances ? "SPOT" : "ON_DEMAND"
      
      # Disk configuration
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type          = "gp3"
            encrypted            = var.enable_encryption
            delete_on_termination = true
          }
        }
      }
      
      # Security
      vpc_security_group_ids = [aws_security_group.eks_additional.id]
      
      # Taints for special workloads
      taints = []
      
      # Labels
      labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
      
      tags = local.common_tags
    }
  }
  
  # Spot instances node group (optional)
  dynamic "eks_managed_node_groups" {
    for_each = var.enable_spot_instances ? ["spot"] : []
    content {
      spot = {
        name           = "${local.cluster_name}-spot"
        instance_types = var.spot_instance_types
        capacity_type  = "SPOT"
        
        min_size     = 0
        max_size     = var.node_group_max_capacity * 2
        desired_size = 1
        
        vpc_security_group_ids = [aws_security_group.eks_additional.id]
        
        labels = {
          Environment = var.environment
          NodeGroup   = "spot"
        }
        
        taints = [
          {
            key    = "spot"
            value  = "true"
            effect = "NO_SCHEDULE"
          }
        ]
        
        tags = merge(local.common_tags, {
          NodeType = "spot"
        })
      }
    }
  }
  
  # Enable IRSA (IAM Roles for Service Accounts)
  enable_irsa = true
  
  tags = local.common_tags
}

# KMS Key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = merge(local.common_tags, {
    Name = "${local.cluster_name}-eks-key"
  })
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.cluster_name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# EKS Cluster autoscaler IAM role
module "cluster_autoscaler_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  
  role_name                        = "${local.cluster_name}-cluster-autoscaler"
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_ids   = [module.eks.cluster_name]
  
  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }
  
  tags = local.common_tags
}

# AWS Load Balancer Controller IAM role
module "load_balancer_controller_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  
  role_name                              = "${local.cluster_name}-aws-load-balancer-controller"
  attach_load_balancer_controller_policy = true
  
  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }
  
  tags = local.common_tags
}

# EBS CSI Driver IAM role
module "ebs_csi_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  
  role_name             = "${local.cluster_name}-ebs-csi"
  attach_ebs_csi_policy = true
  
  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
  
  tags = local.common_tags
}