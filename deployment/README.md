# Production Deployment

This directory contains production deployment configurations for phomem-cosim.

## Files

- `Dockerfile`: Production container image
- `docker-compose.prod.yml`: Docker Compose for production
- `k8s/`: Kubernetes manifests
- `.github/workflows/production.yml`: CI/CD pipeline
- `production-config.json`: Configuration parameters

## Deployment Instructions

### Docker Compose
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

### Monitoring
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/changeme)

## Security Considerations

- Runs as non-root user
- Read-only root filesystem
- Network policies enforced
- Regular security scans in CI/CD

## Scaling

Auto-scaling configured for:
- Min replicas: 3
- Max replicas: 20
- CPU target: 70%
- Memory target: 80%
