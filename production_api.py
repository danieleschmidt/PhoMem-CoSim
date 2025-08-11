#!/usr/bin/env python3
"""
Production FastAPI server for PhoMem-CoSim.
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Import our optimized systems
import sys
sys.path.append('.')
from scalable_optimization_system import ScalableHybridNetwork, OptimizationConfig
from robust_validation_system import RobustHybridNetwork

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
INFERENCE_COUNT = Counter('inferences_total', 'Total inferences')
TRAINING_COUNT = Counter('training_steps_total', 'Total training steps')

# Global network instance
network = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global network
    
    # Startup
    logging.info("Initializing PhoMem-CoSim production server...")
    config = OptimizationConfig(
        enable_caching=True,
        enable_vectorization=True, 
        enable_parallelization=True,
        optimization_level="aggressive"
    )
    network = ScalableHybridNetwork(
        photonic_size=16,
        memristive_rows=16,
        memristive_cols=8,
        config=config
    )
    logging.info("Production server ready")
    yield
    
    # Shutdown
    if network:
        network.cleanup()
    logging.info("Production server shutdown complete")

app = FastAPI(
    title="PhoMem-CoSim API",
    description="Photonic-Memristive Neuromorphic Co-Simulation Platform",
    version="4.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request models
class OpticalInput(BaseModel):
    """Optical input specification."""
    amplitudes: List[float] = Field(..., description="Optical amplitudes")
    phases: List[float] = Field(..., description="Optical phases")
    power_mw: float = Field(1.0, description="Total optical power in mW")

class InferenceRequest(BaseModel):
    """Inference request model."""
    inputs: List[OpticalInput]
    batch_mode: bool = True
    optimization_level: str = "aggressive"

class TrainingRequest(BaseModel):
    """Training request model."""
    training_data: List[Dict[str, Any]]
    epochs: int = 10
    learning_rate: float = 1e-3
    
# Response models
class InferenceResponse(BaseModel):
    """Inference response model."""
    outputs: List[List[float]]
    processing_time: float
    throughput: float
    batch_size: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    version: str = "4.0.0"
    uptime: float

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Request metrics middleware."""
    ACTIVE_CONNECTIONS.inc()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        return response
    finally:
        REQUEST_DURATION.observe(time.time() - start_time)
        ACTIVE_CONNECTIONS.dec()

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        uptime=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if network is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run inference on photonic-memristive network."""
    if network is None:
        raise HTTPException(status_code=503, detail="Network not initialized")
    
    try:
        # Convert inputs to numpy arrays
        import numpy as np
        optical_inputs = []
        for inp in request.inputs:
            if len(inp.amplitudes) != len(inp.phases):
                raise HTTPException(status_code=400, detail="Amplitudes and phases must have same length")
            
            # Convert to complex optical amplitude
            amplitudes = np.array(inp.amplitudes) * np.sqrt(inp.power_mw * 1e-3)
            phases = np.array(inp.phases)
            optical_input = amplitudes * np.exp(1j * phases)
            optical_inputs.append(optical_input)
        
        # Run inference
        result = network.scalable_forward(optical_inputs, batch_mode=request.batch_mode)
        
        # Convert outputs to lists for JSON serialization
        outputs = []
        for output in result["outputs"]:
            if isinstance(output, np.ndarray):
                outputs.append(output.tolist())
            else:
                outputs.append([float(output)])
        
        INFERENCE_COUNT.inc(len(request.inputs))
        
        return InferenceResponse(
            outputs=outputs,
            processing_time=result["processing_time"],
            throughput=result["throughput"],
            batch_size=result["batch_size"]
        )
        
    except Exception as e:
        logging.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/training")
async def run_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Run training on the network."""
    if network is None:
        raise HTTPException(status_code=503, detail="Network not initialized")
    
    try:
        # Convert training data format
        training_data = []
        for item in request.training_data:
            # Expect format: {"input": {...}, "target": [...]}
            optical_input = item["input"]
            target = np.array(item["target"])
            
            amplitudes = np.array(optical_input["amplitudes"])
            phases = np.array(optical_input["phases"])
            optical_array = amplitudes * np.exp(1j * phases)
            
            training_data.append((optical_array, target))
        
        # Run training
        result = network.high_throughput_training(
            training_data, 
            epochs=request.epochs,
            learning_rate=request.learning_rate
        )
        
        TRAINING_COUNT.inc(len(training_data) * request.epochs)
        
        return {
            "status": "completed",
            "training_time": result["training_time"],
            "samples_per_second": result["samples_per_second"],
            "final_loss": result["final_loss"],
            "epochs": request.epochs
        }
        
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/status")
async def system_status():
    """Get comprehensive system status."""
    if network is None:
        raise HTTPException(status_code=503, detail="Network not initialized")
    
    try:
        stats = network.get_comprehensive_stats()
        return {
            "network_status": "operational",
            "optimization_report": stats["optimization_report"],
            "execution_stats": stats["execution_stats"],
            "system_info": stats["system_info"],
            "timestamp": time.time()
        }
    except Exception as e:
        logging.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

if __name__ == "__main__":
    # Production startup
    app.state.start_time = time.time()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use single worker with our optimized threading
        log_level="info",
        access_log=True
    )
