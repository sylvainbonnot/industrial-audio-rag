# Kubernetes and Helm providers
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# AWS Load Balancer Controller
resource "helm_release" "aws_load_balancer_controller" {
  count = var.enable_ingress_controller ? 1 : 0
  
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.6.2"
  
  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }
  
  set {
    name  = "serviceAccount.create"
    value = "true"
  }
  
  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }
  
  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.load_balancer_controller_irsa_role.iam_role_arn
  }
  
  depends_on = [module.eks]
}

# Cluster Autoscaler
resource "helm_release" "cluster_autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  namespace  = "kube-system"
  version    = "9.29.1"
  
  set {
    name  = "autoDiscovery.clusterName"
    value = module.eks.cluster_name
  }
  
  set {
    name  = "awsRegion"
    value = var.aws_region
  }
  
  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.cluster_autoscaler_irsa_role.iam_role_arn
  }
  
  set {
    name  = "extraArgs.scale-down-delay-after-add"
    value = "10m"
  }
  
  set {
    name  = "extraArgs.scale-down-unneeded-time"
    value = "10m"
  }
  
  depends_on = [module.eks]
}

# Prometheus stack (if monitoring enabled)
resource "helm_release" "prometheus" {
  count = var.enable_monitoring ? 1 : 0
  
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  version    = "51.2.0"
  
  create_namespace = true
  
  values = [
    yamlencode({
      prometheus = {
        prometheusSpec = {
          retention = "30d"
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = "gp3"
                accessModes      = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "50Gi"
                  }
                }
              }
            }
          }
        }
      }
      
      grafana = {
        persistence = {
          enabled = true
          storageClassName = "gp3"
          size = "10Gi"
        }
        adminPassword = "admin123"  # Change in production
        ingress = {
          enabled = var.domain_name != ""
          hosts = var.domain_name != "" ? ["grafana.${var.domain_name}"] : []
          tls = var.certificate_arn != "" ? [
            {
              secretName = "grafana-tls"
              hosts = ["grafana.${var.domain_name}"]
            }
          ] : []
        }
      }
      
      alertmanager = {
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = "gp3"
                accessModes      = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "10Gi"
                  }
                }
              }
            }
          }
        }
      }
    })
  ]
  
  depends_on = [module.eks]
}

# NGINX Ingress Controller (alternative to ALB)
resource "helm_release" "nginx_ingress" {
  count = var.enable_ingress_controller ? 1 : 0
  
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress-nginx"
  version    = "4.7.1"
  
  create_namespace = true
  
  set {
    name  = "controller.service.type"
    value = "LoadBalancer"
  }
  
  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-type"
    value = "nlb"
  }
  
  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/aws-load-balancer-cross-zone-load-balancing-enabled"
    value = "true"
  }
  
  depends_on = [module.eks]
}

# Industrial Audio RAG Application
resource "helm_release" "industrial_audio_rag" {
  name      = "industrial-audio-rag"
  chart     = "../../deploy/helm"
  namespace = "default"
  
  values = [
    yamlencode({
      app = {
        image = {
          registry   = "ghcr.io"
          repository = "otosense/industrial-audio-rag"
          tag        = "latest"
        }
        
        env = {
          QDRANT_URL = "http://industrial-audio-rag-qdrant:6333"
        }
      }
      
      qdrant = {
        enabled = true
        persistence = {
          enabled = var.enable_qdrant_persistence
          size    = "${var.qdrant_storage_size}Gi"
        }
      }
      
      ingress = {
        enabled = var.domain_name != ""
        hosts = var.domain_name != "" ? [
          {
            host = var.domain_name
            paths = [
              {
                path     = "/"
                pathType = "Prefix"
              }
            ]
          }
        ] : []
        tls = var.certificate_arn != "" ? [
          {
            secretName = "industrial-audio-rag-tls"
            hosts      = [var.domain_name]
          }
        ] : []
      }
      
      monitoring = {
        prometheus = {
          enabled = var.enable_monitoring
        }
      }
    })
  ]
  
  depends_on = [
    module.eks,
    helm_release.prometheus,
    helm_release.nginx_ingress
  ]
}