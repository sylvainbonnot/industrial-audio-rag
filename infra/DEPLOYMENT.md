# Cloud Deployment Guide

Complete guide for deploying Industrial Audio RAG to AWS using EKS and Terraform.

## 🏗️ Infrastructure Overview

### Architecture Components

- **EKS Cluster**: Managed Kubernetes cluster on AWS
- **VPC**: Isolated network with public/private subnets
- **Node Groups**: Auto-scaling EC2 instances for workloads
- **Application Load Balancer**: Traffic routing and SSL termination
- **Qdrant**: Vector database with persistent storage
- **Monitoring**: Prometheus & Grafana stack
- **Security**: IAM roles, security groups, encryption

### Cost Analysis

| Component | Instance Type | Monthly Cost (USD) | Notes |
|-----------|---------------|-------------------|-------|
| EKS Cluster | Control Plane | $72 | Fixed cost |
| Worker Nodes | 2x t3.medium | $67 | On-demand pricing |
| Load Balancer | NLB | $22.50 | Network Load Balancer |
| EBS Storage | 300GB gp3 | $24 | Node + Qdrant storage |
| **Total** | **Base Setup** | **~$185** | Excludes data transfer |

### Cost Optimization Options

1. **Spot Instances**: Reduce compute costs by 60-90%
2. **Reserved Instances**: 30-75% savings for predictable workloads
3. **Storage Optimization**: Use gp3 instead of gp2, right-size volumes
4. **Auto-scaling**: Scale down during low usage periods

## 🚀 Quick Deployment

### Prerequisites

```bash
# Install required tools
brew install terraform kubectl helm awscli

# Configure AWS credentials
aws configure

# Verify access
aws sts get-caller-identity
```

### 1. Deploy Infrastructure

```bash
# Clone and navigate
cd infra/terraform

# Copy and customize variables
cp terraform.tfvars.example terraform.tfvars
vim terraform.tfvars

# Initialize and deploy
terraform init
terraform plan
terraform apply
```

### 2. Configure kubectl

```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-west-2 --name industrial-audio-rag-dev

# Verify connection
kubectl get nodes
```

### 3. Deploy Application

```bash
# The application is automatically deployed via Terraform
# Check deployment status
kubectl get pods
kubectl get services
kubectl get ingress
```

## 📋 Detailed Configuration

### Environment Variables

Customize these in `terraform.tfvars`:

```hcl
# Production Configuration
environment = "prod"
node_group_desired_capacity = 3
node_group_max_capacity = 10
enable_spot_instances = true

# Domain Configuration
domain_name = "industrial-audio-rag.yourcompany.com"
certificate_arn = "arn:aws:acm:us-west-2:123456789:certificate/abc-123"

# Security
allowed_cidr_blocks = ["10.0.0.0/8", "172.16.0.0/12"]
enable_encryption = true

# Cost Optimization
enable_spot_instances = true
spot_instance_types = ["t3.medium", "t3.large", "m5.large"]
```

### Monitoring Setup

Monitoring is automatically configured when `enable_monitoring = true`:

```bash
# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Get Grafana password
kubectl get secret -n monitoring prometheus-grafana -o jsonpath='{.data.admin-password}' | base64 --decode

# Access Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

### SSL/TLS Configuration

For production deployments with custom domains:

1. **Create ACM Certificate**:
```bash
aws acm request-certificate \
  --domain-name industrial-audio-rag.yourcompany.com \
  --validation-method DNS \
  --region us-west-2
```

2. **Update Terraform variables**:
```hcl
domain_name = "industrial-audio-rag.yourcompany.com"
certificate_arn = "arn:aws:acm:us-west-2:123456789:certificate/your-cert-id"
```

3. **Configure DNS**:
```bash
# Get load balancer hostname
kubectl get ingress industrial-audio-rag

# Create CNAME record pointing to the load balancer
```

## 🔧 Operations

### Scaling

```bash
# Manual scaling
kubectl scale deployment industrial-audio-rag --replicas=5

# Auto-scaling is configured via HPA
kubectl get hpa
```

### Updates

```bash
# Update application image
helm upgrade industrial-audio-rag ./deploy/helm \
  --set app.image.tag=v1.2.0 \
  --namespace default

# Update infrastructure
terraform plan
terraform apply
```

### Monitoring

```bash
# Check application health
kubectl get pods -l app.kubernetes.io/name=industrial-audio-rag

# View logs
kubectl logs -f deployment/industrial-audio-rag

# Check metrics
curl http://localhost:8000/metrics
```

### Backup and Recovery

```bash
# Qdrant data backup (manual process)
kubectl exec -it qdrant-pod -- /bin/bash
# Use Qdrant backup APIs or volume snapshots

# EBS snapshot backup
aws ec2 create-snapshot --volume-id vol-1234567890abcdef0
```

## 🏭 Environment-Specific Configurations

### Development Environment

```hcl
# terraform.tfvars for dev
environment = "dev"
node_group_desired_capacity = 1
node_group_max_capacity = 3
enable_spot_instances = true
enable_monitoring = false
```

### Staging Environment

```hcl
# terraform.tfvars for staging
environment = "staging"
node_group_desired_capacity = 2
node_group_max_capacity = 5
enable_monitoring = true
domain_name = "staging-industrial-audio-rag.yourcompany.com"
```

### Production Environment

```hcl
# terraform.tfvars for prod
environment = "prod"
node_group_desired_capacity = 3
node_group_max_capacity = 10
enable_monitoring = true
enable_encryption = true
backup_retention_days = 30
domain_name = "industrial-audio-rag.yourcompany.com"
certificate_arn = "arn:aws:acm:us-west-2:123456789:certificate/prod-cert"
```

## 🛡️ Security Best Practices

### Network Security

1. **VPC Configuration**: Private subnets for workloads, public subnets for load balancers
2. **Security Groups**: Minimal required access, no wide-open rules
3. **Network Policies**: Kubernetes network policies for pod-to-pod communication

### IAM Security

1. **IRSA**: IAM Roles for Service Accounts for fine-grained permissions
2. **Least Privilege**: Minimal required permissions for each component
3. **Service Accounts**: Dedicated service accounts for each component

### Encryption

1. **At Rest**: EBS volumes and Kubernetes secrets encrypted
2. **In Transit**: TLS for all communications
3. **KMS**: Customer-managed KMS keys for enhanced security

### Access Control

```bash
# RBAC configuration
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
EOF
```

## 📊 Monitoring and Alerting

### Key Metrics to Monitor

1. **Application Metrics**:
   - Request rate and latency
   - Error rates
   - RAG pipeline performance

2. **Infrastructure Metrics**:
   - CPU and memory utilization
   - Disk space and IOPS
   - Network throughput

3. **Kubernetes Metrics**:
   - Pod restart rate
   - Resource requests vs limits
   - Cluster autoscaler events

### Alerting Rules

Prometheus alerting rules are automatically configured for:

- High error rates (>5%)
- High response times (>5s P95)
- Pod crash loops
- Resource exhaustion
- Cluster node failures

### Log Aggregation

```bash
# CloudWatch Container Insights (if enabled)
aws logs describe-log-groups --log-group-name-prefix /aws/containerinsights/

# Access application logs
kubectl logs -f deployment/industrial-audio-rag --tail=100
```

## 🔄 CI/CD Integration

### GitHub Actions Integration

```yaml
# .github/workflows/deploy.yml
- name: Deploy to EKS
  run: |
    aws eks update-kubeconfig --region ${{ env.AWS_REGION }} --name ${{ env.CLUSTER_NAME }}
    helm upgrade --install industrial-audio-rag ./deploy/helm \
      --set app.image.tag=${{ github.sha }} \
      --wait --timeout=5m
```

### Blue-Green Deployment

```bash
# Deploy to new environment
terraform workspace new green
terraform apply -var="environment=green"

# Switch traffic
kubectl patch ingress industrial-audio-rag -p '{"spec":{"rules":[{"host":"yourapp.com","http":{"paths":[{"path":"/","backend":{"service":{"name":"industrial-audio-rag-green","port":{"number":8000}}}}]}}]}}'
```

## 🔍 Troubleshooting

### Common Issues

1. **Pod Startup Issues**:
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name> --previous
```

2. **Storage Issues**:
```bash
kubectl get pv,pvc
kubectl describe pvc qdrant-storage
```

3. **Network Issues**:
```bash
kubectl get svc,endpoints
kubectl describe ingress industrial-audio-rag
```

4. **Resource Issues**:
```bash
kubectl top nodes
kubectl top pods
kubectl get events --sort-by='.lastTimestamp'
```

### Performance Tuning

1. **Resource Requests/Limits**:
```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

2. **JVM Tuning** (for Java components):
```yaml
env:
  - name: JAVA_OPTS
    value: "-Xms2g -Xmx4g -XX:+UseG1GC"
```

3. **Database Optimization**:
```yaml
# Qdrant configuration
env:
  - name: QDRANT_GRPC_PORT
    value: "6334"
  - name: QDRANT_HTTP_PORT
    value: "6333"
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Review monitoring dashboards and alerts
2. **Monthly**: Update cluster and node group versions
3. **Quarterly**: Review and optimize costs
4. **As needed**: Security patches and application updates

### Emergency Procedures

1. **Application Down**:
   - Check pod status and logs
   - Verify ingress and service configuration
   - Scale up replicas if needed

2. **Database Issues**:
   - Check Qdrant pod status
   - Verify persistent volume claims
   - Restore from backup if necessary

3. **Cluster Issues**:
   - Check node status
   - Review cluster autoscaler logs
   - Contact AWS support if needed

### Cleanup

```bash
# Remove application
helm uninstall industrial-audio-rag

# Destroy infrastructure
terraform destroy

# Clean up local state
terraform workspace delete dev
```

This deployment guide provides a production-ready foundation for running Industrial Audio RAG at scale on AWS.