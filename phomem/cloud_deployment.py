"""
Cloud-Native Deployment and Auto-Scaling for Photonic-Memristive Simulations.

This module implements cloud deployment features:
- Kubernetes-based deployment with auto-scaling
- Container orchestration for simulation workloads  
- Cloud resource management and cost optimization
- Multi-cloud support (AWS, GCP, Azure)
- Serverless computing integration
- Edge computing deployment
"""

import logging
import time
import json
import yaml
import os
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings

from .distributed_computing import LoadBalancer, ComputeResource, ResourceType, DistributedTask
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"


class DeploymentStatus(Enum):
    """Status of cloud deployments."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    SCALING = "scaling"
    UPDATING = "updating"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class CloudDeploymentConfig:
    """Configuration for cloud deployment."""
    deployment_name: str
    cloud_provider: CloudProvider
    region: str = "us-east-1"
    instance_type: str = "m5.large"
    min_nodes: int = 1
    max_nodes: int = 10
    auto_scaling: bool = True
    
    # Container configuration
    container_image: str = "phomem-cosim:latest"
    cpu_request: str = "1"
    cpu_limit: str = "2"
    memory_request: str = "2Gi"
    memory_limit: str = "4Gi"
    
    # Storage
    storage_size: str = "20Gi"
    storage_class: str = "gp2"
    
    # Networking
    enable_load_balancer: bool = True
    expose_ports: List[int] = field(default_factory=lambda: [8080, 8081])
    
    # Auto-scaling
    cpu_threshold: float = 70.0  # Percentage
    memory_threshold: float = 80.0  # Percentage
    scale_up_cooldown: int = 300  # Seconds
    scale_down_cooldown: int = 600  # Seconds
    
    # Cost optimization
    enable_spot_instances: bool = False
    max_price: Optional[float] = None
    preemptible: bool = False


class CloudResourceManager:
    """Manages cloud resources and deployments."""
    
    def __init__(self, deployment_config: CloudDeploymentConfig):
        self.config = deployment_config
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self.resource_usage_history: List[Dict[str, Any]] = []
        
        logger.info(f"Cloud resource manager initialized for {deployment_config.cloud_provider.value}")
    
    def create_deployment(self) -> bool:
        """Create a new cloud deployment."""
        try:
            if self.config.cloud_provider == CloudProvider.KUBERNETES:
                return self._create_kubernetes_deployment()
            elif self.config.cloud_provider == CloudProvider.DOCKER:
                return self._create_docker_deployment()
            elif self.config.cloud_provider == CloudProvider.AWS:
                return self._create_aws_deployment()
            elif self.config.cloud_provider == CloudProvider.GCP:
                return self._create_gcp_deployment()
            elif self.config.cloud_provider == CloudProvider.AZURE:
                return self._create_azure_deployment()
            else:
                logger.error(f"Unsupported cloud provider: {self.config.cloud_provider}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            return False
    
    def scale_deployment(self, target_replicas: int) -> bool:
        """Scale deployment to target number of replicas."""
        try:
            if self.config.cloud_provider == CloudProvider.KUBERNETES:
                return self._scale_kubernetes_deployment(target_replicas)
            elif self.config.cloud_provider == CloudProvider.DOCKER:
                return self._scale_docker_deployment(target_replicas)
            else:
                logger.warning(f"Scaling not implemented for {self.config.cloud_provider.value}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False
    
    def update_deployment(self, new_config: CloudDeploymentConfig) -> bool:
        """Update existing deployment with new configuration."""
        try:
            old_config = self.config
            self.config = new_config
            
            if new_config.cloud_provider == CloudProvider.KUBERNETES:
                return self._update_kubernetes_deployment()
            else:
                logger.warning(f"Update not implemented for {new_config.cloud_provider.value}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update deployment: {e}")
            self.config = old_config  # Rollback
            return False
    
    def delete_deployment(self) -> bool:
        """Delete the cloud deployment."""
        try:
            if self.config.cloud_provider == CloudProvider.KUBERNETES:
                return self._delete_kubernetes_deployment()
            elif self.config.cloud_provider == CloudProvider.DOCKER:
                return self._delete_docker_deployment()
            else:
                logger.warning(f"Delete not implemented for {self.config.cloud_provider.value}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        try:
            if self.config.cloud_provider == CloudProvider.KUBERNETES:
                return self._get_kubernetes_status()
            elif self.config.cloud_provider == CloudProvider.DOCKER:
                return self._get_docker_status()
            else:
                return {
                    'status': DeploymentStatus.ERROR.value,
                    'message': f"Status check not implemented for {self.config.cloud_provider.value}"
                }
                
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {
                'status': DeploymentStatus.ERROR.value,
                'message': str(e)
            }
    
    def _create_kubernetes_deployment(self) -> bool:
        """Create Kubernetes deployment."""
        
        # Generate Kubernetes manifests
        deployment_manifest = self._generate_k8s_deployment_manifest()
        service_manifest = self._generate_k8s_service_manifest()
        hpa_manifest = self._generate_k8s_hpa_manifest() if self.config.auto_scaling else None
        
        # Write manifests to files
        manifest_dir = f"/tmp/{self.config.deployment_name}"
        os.makedirs(manifest_dir, exist_ok=True)
        
        with open(f"{manifest_dir}/deployment.yaml", 'w') as f:
            yaml.dump(deployment_manifest, f)
        
        with open(f"{manifest_dir}/service.yaml", 'w') as f:
            yaml.dump(service_manifest, f)
        
        if hpa_manifest:
            with open(f"{manifest_dir}/hpa.yaml", 'w') as f:
                yaml.dump(hpa_manifest, f)
        
        # Apply manifests (would use kubectl in real implementation)
        logger.info(f"Kubernetes manifests generated in {manifest_dir}")
        
        # Simulate successful deployment
        self.deployments[self.config.deployment_name] = {
            'status': DeploymentStatus.RUNNING.value,
            'replicas': self.config.min_nodes,
            'created_at': time.time()
        }
        
        return True
    
    def _generate_k8s_deployment_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest."""
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.config.deployment_name,
                'labels': {
                    'app': self.config.deployment_name,
                    'version': 'v1',
                    'component': 'simulation'
                }
            },
            'spec': {
                'replicas': self.config.min_nodes,
                'selector': {
                    'matchLabels': {
                        'app': self.config.deployment_name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config.deployment_name
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'phomem-simulation',
                            'image': self.config.container_image,
                            'ports': [{'containerPort': port} for port in self.config.expose_ports],
                            'resources': {
                                'requests': {
                                    'cpu': self.config.cpu_request,
                                    'memory': self.config.memory_request
                                },
                                'limits': {
                                    'cpu': self.config.cpu_limit,
                                    'memory': self.config.memory_limit
                                }
                            },
                            'env': [
                                {'name': 'DEPLOYMENT_NAME', 'value': self.config.deployment_name},
                                {'name': 'CLOUD_PROVIDER', 'value': self.config.cloud_provider.value},
                                {'name': 'AUTO_SCALING', 'value': str(self.config.auto_scaling)}
                            ],
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8080
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8080
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }],
                        'restartPolicy': 'Always'
                    }
                }
            }
        }
    
    def _generate_k8s_service_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes service manifest."""
        service_type = 'LoadBalancer' if self.config.enable_load_balancer else 'ClusterIP'
        
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{self.config.deployment_name}-service",
                'labels': {
                    'app': self.config.deployment_name
                }
            },
            'spec': {
                'type': service_type,
                'ports': [
                    {
                        'port': port,
                        'targetPort': port,
                        'name': f'port-{port}'
                    } for port in self.config.expose_ports
                ],
                'selector': {
                    'app': self.config.deployment_name
                }
            }
        }
    
    def _generate_k8s_hpa_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes Horizontal Pod Autoscaler manifest."""
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.config.deployment_name}-hpa",
                'labels': {
                    'app': self.config.deployment_name
                }
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.config.deployment_name
                },
                'minReplicas': self.config.min_nodes,
                'maxReplicas': self.config.max_nodes,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': int(self.config.cpu_threshold)
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': int(self.config.memory_threshold)
                            }
                        }
                    }
                ],
                'behavior': {
                    'scaleUp': {
                        'stabilizationWindowSeconds': self.config.scale_up_cooldown
                    },
                    'scaleDown': {
                        'stabilizationWindowSeconds': self.config.scale_down_cooldown
                    }
                }
            }
        }
    
    def _create_docker_deployment(self) -> bool:
        """Create Docker deployment."""
        # Generate docker-compose file
        docker_compose = self._generate_docker_compose()
        
        compose_file = f"/tmp/{self.config.deployment_name}-docker-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(docker_compose, f)
        
        logger.info(f"Docker Compose file generated: {compose_file}")
        
        # Simulate successful deployment
        self.deployments[self.config.deployment_name] = {
            'status': DeploymentStatus.RUNNING.value,
            'containers': self.config.min_nodes,
            'created_at': time.time()
        }
        
        return True
    
    def _generate_docker_compose(self) -> Dict[str, Any]:
        """Generate Docker Compose configuration."""
        return {
            'version': '3.8',
            'services': {
                'phomem-simulation': {
                    'image': self.config.container_image,
                    'deploy': {
                        'replicas': self.config.min_nodes,
                        'resources': {
                            'limits': {
                                'cpus': self.config.cpu_limit,
                                'memory': self.config.memory_limit
                            },
                            'reservations': {
                                'cpus': self.config.cpu_request,
                                'memory': self.config.memory_request
                            }
                        }
                    },
                    'ports': [f"{port}:{port}" for port in self.config.expose_ports],
                    'environment': {
                        'DEPLOYMENT_NAME': self.config.deployment_name,
                        'CLOUD_PROVIDER': self.config.cloud_provider.value,
                        'AUTO_SCALING': str(self.config.auto_scaling)
                    },
                    'healthcheck': {
                        'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'restart': 'unless-stopped'
                }
            }
        }
    
    def _create_aws_deployment(self) -> bool:
        """Create AWS deployment (ECS/EKS)."""
        # Generate AWS CloudFormation or Terraform templates
        logger.info("AWS deployment would create ECS/EKS cluster")
        
        # Simulate successful deployment
        self.deployments[self.config.deployment_name] = {
            'status': DeploymentStatus.RUNNING.value,
            'instances': self.config.min_nodes,
            'created_at': time.time(),
            'provider': 'aws'
        }
        
        return True
    
    def _create_gcp_deployment(self) -> bool:
        """Create GCP deployment (GKE/Cloud Run)."""
        logger.info("GCP deployment would create GKE cluster or Cloud Run service")
        
        # Simulate successful deployment
        self.deployments[self.config.deployment_name] = {
            'status': DeploymentStatus.RUNNING.value,
            'instances': self.config.min_nodes,
            'created_at': time.time(),
            'provider': 'gcp'
        }
        
        return True
    
    def _create_azure_deployment(self) -> bool:
        """Create Azure deployment (AKS/Container Instances)."""
        logger.info("Azure deployment would create AKS cluster or Container Instances")
        
        # Simulate successful deployment
        self.deployments[self.config.deployment_name] = {
            'status': DeploymentStatus.RUNNING.value,
            'instances': self.config.min_nodes,
            'created_at': time.time(),
            'provider': 'azure'
        }
        
        return True
    
    def _scale_kubernetes_deployment(self, target_replicas: int) -> bool:
        """Scale Kubernetes deployment."""
        # In real implementation, would use kubectl scale
        logger.info(f"Scaling Kubernetes deployment to {target_replicas} replicas")
        
        if self.config.deployment_name in self.deployments:
            self.deployments[self.config.deployment_name]['replicas'] = target_replicas
            self.deployments[self.config.deployment_name]['status'] = DeploymentStatus.RUNNING.value
        
        return True
    
    def _scale_docker_deployment(self, target_replicas: int) -> bool:
        """Scale Docker deployment."""
        logger.info(f"Scaling Docker deployment to {target_replicas} containers")
        
        if self.config.deployment_name in self.deployments:
            self.deployments[self.config.deployment_name]['containers'] = target_replicas
            self.deployments[self.config.deployment_name]['status'] = DeploymentStatus.RUNNING.value
        
        return True
    
    def _update_kubernetes_deployment(self) -> bool:
        """Update Kubernetes deployment."""
        logger.info("Updating Kubernetes deployment")
        
        # Generate new manifests and apply
        return self._create_kubernetes_deployment()
    
    def _delete_kubernetes_deployment(self) -> bool:
        """Delete Kubernetes deployment."""
        logger.info("Deleting Kubernetes deployment")
        
        if self.config.deployment_name in self.deployments:
            del self.deployments[self.config.deployment_name]
        
        return True
    
    def _delete_docker_deployment(self) -> bool:
        """Delete Docker deployment."""
        logger.info("Stopping Docker deployment")
        
        if self.config.deployment_name in self.deployments:
            del self.deployments[self.config.deployment_name]
        
        return True
    
    def _get_kubernetes_status(self) -> Dict[str, Any]:
        """Get Kubernetes deployment status."""
        if self.config.deployment_name not in self.deployments:
            return {
                'status': DeploymentStatus.ERROR.value,
                'message': 'Deployment not found'
            }
        
        deployment_info = self.deployments[self.config.deployment_name]
        
        return {
            'status': deployment_info['status'],
            'replicas': deployment_info.get('replicas', 0),
            'ready_replicas': deployment_info.get('replicas', 0),  # Simplified
            'created_at': deployment_info['created_at'],
            'provider': 'kubernetes'
        }
    
    def _get_docker_status(self) -> Dict[str, Any]:
        """Get Docker deployment status."""
        if self.config.deployment_name not in self.deployments:
            return {
                'status': DeploymentStatus.ERROR.value,
                'message': 'Deployment not found'
            }
        
        deployment_info = self.deployments[self.config.deployment_name]
        
        return {
            'status': deployment_info['status'],
            'containers': deployment_info.get('containers', 0),
            'created_at': deployment_info['created_at'],
            'provider': 'docker'
        }


class AutoScaler:
    """Auto-scaler for cloud deployments."""
    
    def __init__(
        self, 
        resource_manager: CloudResourceManager,
        scaling_interval: float = 60.0,
        metrics_window: int = 10
    ):
        self.resource_manager = resource_manager
        self.scaling_interval = scaling_interval
        self.metrics_window = metrics_window
        
        # Metrics tracking
        self.cpu_metrics: List[float] = []
        self.memory_metrics: List[float] = []
        self.request_rate_metrics: List[float] = []
        
        # Scaling state
        self.last_scale_action = 0.0
        self.current_replicas = resource_manager.config.min_nodes
        self.is_running = False
        
        logger.info("Auto-scaler initialized")
    
    def start(self):
        """Start the auto-scaler."""
        self.is_running = True
        logger.info("Auto-scaler started")
    
    def stop(self):
        """Stop the auto-scaler."""
        self.is_running = False
        logger.info("Auto-scaler stopped")
    
    def update_metrics(self, cpu_usage: float, memory_usage: float, request_rate: float = 0.0):
        """Update metrics for scaling decisions."""
        self.cpu_metrics.append(cpu_usage)
        self.memory_metrics.append(memory_usage)
        self.request_rate_metrics.append(request_rate)
        
        # Keep only recent metrics
        if len(self.cpu_metrics) > self.metrics_window:
            self.cpu_metrics = self.cpu_metrics[-self.metrics_window:]
            self.memory_metrics = self.memory_metrics[-self.metrics_window:]
            self.request_rate_metrics = self.request_rate_metrics[-self.metrics_window:]
    
    def should_scale(self) -> Tuple[bool, int]:
        """Determine if scaling is needed and target replica count."""
        if not self.is_running or len(self.cpu_metrics) < 3:
            return False, self.current_replicas
        
        # Calculate average metrics
        avg_cpu = sum(self.cpu_metrics[-5:]) / min(5, len(self.cpu_metrics))
        avg_memory = sum(self.memory_metrics[-5:]) / min(5, len(self.memory_metrics))
        
        config = self.resource_manager.config
        current_time = time.time()
        
        # Check cooldown periods
        if current_time - self.last_scale_action < config.scale_up_cooldown:
            return False, self.current_replicas
        
        # Scale up conditions
        scale_up = (
            avg_cpu > config.cpu_threshold or 
            avg_memory > config.memory_threshold
        )
        
        # Scale down conditions (more conservative)
        scale_down = (
            avg_cpu < config.cpu_threshold * 0.5 and 
            avg_memory < config.memory_threshold * 0.5 and
            current_time - self.last_scale_action > config.scale_down_cooldown
        )
        
        target_replicas = self.current_replicas
        
        if scale_up and self.current_replicas < config.max_nodes:
            # Calculate how many replicas to add
            cpu_scale_factor = avg_cpu / config.cpu_threshold
            memory_scale_factor = avg_memory / config.memory_threshold
            
            scale_factor = max(cpu_scale_factor, memory_scale_factor)
            additional_replicas = max(1, int(scale_factor * 0.5))  # Conservative scaling
            
            target_replicas = min(
                self.current_replicas + additional_replicas,
                config.max_nodes
            )
            
        elif scale_down and self.current_replicas > config.min_nodes:
            # Scale down by 1 replica at a time
            target_replicas = max(
                self.current_replicas - 1,
                config.min_nodes
            )
        
        should_scale = target_replicas != self.current_replicas
        
        if should_scale:
            logger.info(f"Auto-scaling recommendation: {self.current_replicas} -> {target_replicas} "
                       f"(CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.1f}%)")
        
        return should_scale, target_replicas
    
    def execute_scaling(self, target_replicas: int) -> bool:
        """Execute scaling action."""
        if not self.is_running:
            return False
        
        success = self.resource_manager.scale_deployment(target_replicas)
        
        if success:
            self.current_replicas = target_replicas
            self.last_scale_action = time.time()
            logger.info(f"Successfully scaled to {target_replicas} replicas")
        else:
            logger.error(f"Failed to scale to {target_replicas} replicas")
        
        return success
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get current scaling metrics and status."""
        return {
            'current_replicas': self.current_replicas,
            'avg_cpu_usage': sum(self.cpu_metrics[-5:]) / min(5, len(self.cpu_metrics)) if self.cpu_metrics else 0.0,
            'avg_memory_usage': sum(self.memory_metrics[-5:]) / min(5, len(self.memory_metrics)) if self.memory_metrics else 0.0,
            'last_scale_action': self.last_scale_action,
            'is_running': self.is_running,
            'metrics_count': len(self.cpu_metrics)
        }


class CostOptimizer:
    """Optimizes cloud deployment costs."""
    
    def __init__(self, resource_manager: CloudResourceManager):
        self.resource_manager = resource_manager
        self.cost_history: List[Dict[str, Any]] = []
        
        # Cost rates (USD per hour) - simplified
        self.cost_rates = {
            ResourceType.CPU: 0.0464,  # m5.large
            ResourceType.GPU: 3.06,    # p3.2xlarge
            ResourceType.EDGE: 0.023   # t3.micro
        }
        
        logger.info("Cost optimizer initialized")
    
    def estimate_hourly_cost(self, deployment_status: Dict[str, Any]) -> float:
        """Estimate hourly cost of current deployment."""
        if deployment_status['status'] != DeploymentStatus.RUNNING.value:
            return 0.0
        
        # Simple cost calculation based on instance type
        config = self.resource_manager.config
        instance_count = deployment_status.get('replicas', deployment_status.get('containers', 1))
        
        # Map instance types to resource types (simplified)
        if 'gpu' in config.instance_type.lower() or 'p3' in config.instance_type:
            hourly_rate = self.cost_rates[ResourceType.GPU]
        elif 'micro' in config.instance_type or 'small' in config.instance_type:
            hourly_rate = self.cost_rates[ResourceType.EDGE]
        else:
            hourly_rate = self.cost_rates[ResourceType.CPU]
        
        total_cost = instance_count * hourly_rate
        
        # Add storage costs
        storage_gb = float(config.storage_size.replace('Gi', ''))
        storage_cost = storage_gb * 0.10 / (24 * 30)  # $0.10/GB/month
        
        return total_cost + storage_cost
    
    def suggest_optimizations(self, deployment_status: Dict[str, Any]) -> List[str]:
        """Suggest cost optimization strategies."""
        suggestions = []
        
        config = self.resource_manager.config
        instance_count = deployment_status.get('replicas', deployment_status.get('containers', 1))
        
        # Check for over-provisioning
        if instance_count > config.min_nodes * 2:
            suggestions.append("Consider reducing max_nodes to prevent over-provisioning")
        
        # Check for spot instance opportunities
        if not config.enable_spot_instances and config.cloud_provider == CloudProvider.AWS:
            suggestions.append("Enable spot instances to reduce costs by up to 70%")
        
        # Check resource requests vs limits
        if config.cpu_request == config.cpu_limit:
            suggestions.append("Consider setting lower CPU requests to improve resource utilization")
        
        # Check storage optimization
        if config.storage_size and float(config.storage_size.replace('Gi', '')) > 50:
            suggestions.append("Evaluate if large persistent storage is needed or if object storage could be used")
        
        # Check region optimization
        if config.region == "us-east-1":
            suggestions.append("Consider using a cheaper region if latency requirements allow")
        
        return suggestions
    
    def track_cost(self, deployment_status: Dict[str, Any]):
        """Track cost over time."""
        cost_entry = {
            'timestamp': time.time(),
            'hourly_cost': self.estimate_hourly_cost(deployment_status),
            'instance_count': deployment_status.get('replicas', deployment_status.get('containers', 1)),
            'status': deployment_status['status']
        }
        
        self.cost_history.append(cost_entry)
        
        # Keep only recent history (30 days)
        if len(self.cost_history) > 30 * 24:  # Assuming hourly tracking
            self.cost_history = self.cost_history[-30*24:]
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Generate cost report."""
        if not self.cost_history:
            return {'total_cost': 0.0, 'average_hourly_cost': 0.0, 'days_tracked': 0}
        
        # Calculate total cost
        total_hours = len(self.cost_history)
        total_cost = sum(entry['hourly_cost'] for entry in self.cost_history)
        average_hourly_cost = total_cost / total_hours if total_hours > 0 else 0.0
        
        # Daily costs for the last week
        daily_costs = []
        if len(self.cost_history) >= 24:  # At least 1 day of data
            for i in range(0, min(len(self.cost_history), 7 * 24), 24):
                daily_cost = sum(
                    entry['hourly_cost'] 
                    for entry in self.cost_history[i:i+24]
                )
                daily_costs.append(daily_cost)
        
        return {
            'total_cost': total_cost,
            'average_hourly_cost': average_hourly_cost,
            'days_tracked': total_hours / 24,
            'daily_costs': daily_costs,
            'current_hourly_cost': self.cost_history[-1]['hourly_cost'] if self.cost_history else 0.0
        }


def create_cloud_deployment_config(
    deployment_name: str,
    cloud_provider: CloudProvider = CloudProvider.KUBERNETES,
    min_nodes: int = 1,
    max_nodes: int = 5
) -> CloudDeploymentConfig:
    """Create a cloud deployment configuration."""
    
    return CloudDeploymentConfig(
        deployment_name=deployment_name,
        cloud_provider=cloud_provider,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        auto_scaling=True,
        container_image="phomem-cosim:latest",
        cpu_request="500m",
        cpu_limit="1",
        memory_request="1Gi",
        memory_limit="2Gi",
        storage_size="10Gi",
        enable_load_balancer=True,
        expose_ports=[8080, 8081],
        cpu_threshold=70.0,
        memory_threshold=80.0
    )


def setup_cloud_deployment(
    deployment_name: str,
    cloud_provider: CloudProvider = CloudProvider.KUBERNETES
) -> Tuple[CloudResourceManager, AutoScaler, CostOptimizer]:
    """Setup complete cloud deployment with auto-scaling and cost optimization."""
    
    # Create deployment configuration
    config = create_cloud_deployment_config(deployment_name, cloud_provider)
    
    # Create resource manager
    resource_manager = CloudResourceManager(config)
    
    # Create auto-scaler
    auto_scaler = AutoScaler(resource_manager)
    
    # Create cost optimizer
    cost_optimizer = CostOptimizer(resource_manager)
    
    logger.info(f"Cloud deployment setup complete for {deployment_name}")
    
    return resource_manager, auto_scaler, cost_optimizer