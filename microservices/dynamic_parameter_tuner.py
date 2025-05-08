"""Dynamic Parameter Tuner implementation.

This service optimizes retrieval parameters based on historical performance data.
It tracks query performance, records feedback, and updates parameters to improve
results over time.
"""
import asyncio
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx
import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

from microservices.service_interfaces import (
    ServiceRequest,
    ServiceResponse
)

# Configure logging
logger = logging.getLogger(__name__)


class OptimizeParametersRequest(ServiceRequest):
    """Optimize parameters request model."""
    query_type: str
    query_features: Dict[str, Any]
    query_text: Optional[str] = None


class RecordFeedbackRequest(ServiceRequest):
    """Record feedback request model."""
    query_id: str
    parameters_used: Dict[str, Any]
    performance_metrics: Dict[str, float]
    query_type: str
    query_features: Dict[str, Any]


class TrainModelRequest(ServiceRequest):
    """Train parameter model request."""
    query_type: Optional[str] = None  # If None, train models for all query types


class ParameterResponse(ServiceResponse):
    """Parameter tuner response model."""
    optimized_parameters: Optional[Dict[str, Any]] = None
    model_info: Optional[Dict[str, Any]] = None


class DynamicParameterTuner:
    """Dynamic Parameter Tuner for optimizing retrieval parameters."""
    
    def __init__(
        self,
        model_dir: str = "parameter_models",
        history_file: str = "performance_history.json",
        cache_service_url: Optional[str] = None,
        min_feedback_samples: int = 10,
        update_interval: int = 24  # Hours
    ):
        """Initialize the parameter tuner.
        
        Args:
            model_dir: Directory to store parameter models
            history_file: File to store performance history
            cache_service_url: Optional URL for the Cache Service
            min_feedback_samples: Minimum feedback samples before training
            update_interval: Hours between model updates
        """
        self.model_dir = model_dir
        self.history_file = os.path.join(model_dir, history_file)
        self.cache_service_url = cache_service_url
        self.min_feedback_samples = min_feedback_samples
        self.update_interval = update_interval * 3600  # Convert to seconds
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize parameter models
        self.parameter_models = {}
        self.scalers = {}
        self.default_parameters = self._get_default_parameters()
        
        # Performance history
        self.performance_history = []
        self._load_history()
        
        # Load existing models
        self._load_models()
        
        # Tracking variables
        self.last_update_time = time.time()
        self.pending_updates = defaultdict(int)
        
        # Initialize httpx client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        
        logger.info(f"Dynamic Parameter Tuner initialized")
    
    def _get_default_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get default parameters for different query types."""
        return {
            "factual": {
                "alpha": 0.3,  # Weight for vector search (vs. BM25)
                "fusion_method": "rrf",
                "use_colbert": False,
                "use_splade": True,
                "rerank_mmr_lambda": 0.7,
                "diversity_weight": 0.1
            },
            "conceptual": {
                "alpha": 0.7,
                "fusion_method": "softmax",
                "use_colbert": True,
                "use_splade": False,
                "rerank_mmr_lambda": 0.5,
                "diversity_weight": 0.3
            },
            "balanced": {
                "alpha": 0.5,
                "fusion_method": "linear",
                "use_colbert": True,
                "use_splade": True,
                "rerank_mmr_lambda": 0.6,
                "diversity_weight": 0.2
            },
            "entity_rich": {
                "alpha": 0.4,
                "fusion_method": "rrf",
                "use_colbert": False,
                "use_splade": True,
                "rerank_mmr_lambda": 0.8,
                "diversity_weight": 0.1
            },
            "complex": {
                "alpha": 0.6,
                "fusion_method": "softmax",
                "use_colbert": True,
                "use_splade": True,
                "rerank_mmr_lambda": 0.5,
                "diversity_weight": 0.4
            }
        }
    
    def _load_history(self):
        """Load performance history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.performance_history = json.load(f)
                logger.info(f"Loaded {len(self.performance_history)} performance history entries")
            except Exception as e:
                logger.error(f"Error loading performance history: {str(e)}")
                self.performance_history = []
    
    def _save_history(self):
        """Save performance history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.performance_history, f)
            logger.info(f"Saved {len(self.performance_history)} performance history entries")
        except Exception as e:
            logger.error(f"Error saving performance history: {str(e)}")
    
    def _load_models(self):
        """Load parameter models from disk."""
        for query_type in self.default_parameters.keys():
            model_path = os.path.join(self.model_dir, f"{query_type}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{query_type}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.parameter_models[query_type] = pickle.load(f)
                    
                    with open(scaler_path, 'rb') as f:
                        self.scalers[query_type] = pickle.load(f)
                    
                    logger.info(f"Loaded parameter model for query type: {query_type}")
                except Exception as e:
                    logger.error(f"Error loading parameter model for {query_type}: {str(e)}")
    
    def _save_model(self, query_type: str):
        """Save parameter model to disk.
        
        Args:
            query_type: Query type for the model
        """
        if query_type in self.parameter_models:
            model_path = os.path.join(self.model_dir, f"{query_type}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{query_type}_scaler.pkl")
            
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(self.parameter_models[query_type], f)
                
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[query_type], f)
                
                logger.info(f"Saved parameter model for query type: {query_type}")
            except Exception as e:
                logger.error(f"Error saving parameter model for {query_type}: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        status = {
            "status": "healthy",
            "version": "1.0.0",
            "trained_query_types": list(self.parameter_models.keys()),
            "feedback_entries": len(self.performance_history),
            "last_update_time": self.last_update_time
        }
        
        # Check cache service if configured
        if self.cache_service_url:
            try:
                response = await self.client.get(f"{self.cache_service_url}/health")
                status["cache_status"] = "connected" if response.status_code == 200 else "error"
            except Exception as e:
                status["cache_status"] = f"error: {str(e)}"
        
        return status
    
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request and return a response."""
        if isinstance(request, OptimizeParametersRequest):
            try:
                parameters = await self.optimize_parameters(
                    query_type=request.query_type,
                    query_features=request.query_features,
                    query_text=request.query_text
                )
                
                return ParameterResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"optimized_parameters": parameters}
                )
                
            except Exception as e:
                logger.error(f"Error optimizing parameters: {str(e)}")
                return ParameterResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error optimizing parameters: {str(e)}"
                )
                
        elif isinstance(request, RecordFeedbackRequest):
            try:
                await self.record_feedback(
                    query_id=request.query_id,
                    parameters_used=request.parameters_used,
                    performance_metrics=request.performance_metrics,
                    query_type=request.query_type,
                    query_features=request.query_features
                )
                
                return ServiceResponse(
                    request_id=request.request_id,
                    status="success",
                    message="Feedback recorded successfully"
                )
                
            except Exception as e:
                logger.error(f"Error recording feedback: {str(e)}")
                return ServiceResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error recording feedback: {str(e)}"
                )
                
        elif isinstance(request, TrainModelRequest):
            try:
                model_info = await self.train_model(request.query_type)
                
                return ParameterResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"model_info": model_info}
                )
                
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
                return ParameterResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error training model: {str(e)}"
                )
                
        else:
            return ServiceResponse(
                request_id=getattr(request, "request_id", "unknown"),
                status="error",
                message="Invalid request type"
            )
    
    async def optimize_parameters(
        self, 
        query_type: str, 
        query_features: Dict[str, Any],
        query_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize parameters for a specific query.
        
        Args:
            query_type: Type of query (e.g., factual, conceptual, balanced)
            query_features: Features of the query for parameter optimization
            query_text: Optional query text for additional analysis
            
        Returns:
            Dictionary of optimized parameters
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            
            # Generate a deterministic representation of inputs for caching
            cache_input = {
                "query_type": query_type,
                "query_features": query_features
            }
            
            if query_text:
                cache_input["query_text"] = query_text
            
            cache_str = json.dumps(cache_input, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
            cache_key = f"parameters:{cache_hash}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for parameter optimization")
                        return cached_data
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # Check if we need to update models
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            # Check if we have enough new feedback to warrant an update
            for query_type, count in self.pending_updates.items():
                if count >= self.min_feedback_samples:
                    await self.train_model(query_type)
            
            self.last_update_time = current_time
            self.pending_updates.clear()
        
        # If no model exists for this query type, use default parameters
        if query_type not in self.parameter_models:
            if query_type in self.default_parameters:
                return self.default_parameters[query_type].copy()
            else:
                # Fallback to balanced
                return self.default_parameters["balanced"].copy()
        
        # Extract features for model
        feature_vector = self._extract_feature_vector(query_features)
        
        # Apply feature scaling
        scaled_features = self.scalers[query_type].transform([feature_vector])[0]
        
        # Get model for this query type
        model = self.parameter_models[query_type]
        
        # Get default parameters for this query type
        parameters = self.default_parameters[query_type].copy()
        
        # Update continuous parameters using the model
        try:
            # Predict alpha (vector weight)
            parameters["alpha"] = float(model["alpha"].predict([scaled_features])[0])
            parameters["alpha"] = max(0.1, min(0.9, parameters["alpha"]))  # Clamp to valid range
            
            # Predict MMR lambda
            parameters["rerank_mmr_lambda"] = float(model["rerank_mmr_lambda"].predict([scaled_features])[0])
            parameters["rerank_mmr_lambda"] = max(0.1, min(0.9, parameters["rerank_mmr_lambda"]))
            
            # Predict diversity weight
            parameters["diversity_weight"] = float(model["diversity_weight"].predict([scaled_features])[0])
            parameters["diversity_weight"] = max(0, min(0.5, parameters["diversity_weight"]))
            
            # Apply query-specific optimizations
            if "entity_count" in query_features and query_features["entity_count"] > 2:
                parameters["use_splade"] = True
            
            if "word_count" in query_features and query_features["word_count"] > 15:
                parameters["use_colbert"] = True
                
        except Exception as e:
            logger.error(f"Error predicting parameters: {str(e)}")
            # Return default parameters in case of error
            return self.default_parameters[query_type].copy()
        
        # Cache results if cache service is configured
        if self.cache_service_url:
            import hashlib
            
            cache_input = {
                "query_type": query_type,
                "query_features": query_features
            }
            
            if query_text:
                cache_input["query_text"] = query_text
            
            cache_str = json.dumps(cache_input, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
            cache_key = f"parameters:{cache_hash}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": parameters,
                        "ttl": 3600  # 1 hour TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching parameters: {str(e)}")
        
        return parameters
    
    async def record_feedback(
        self,
        query_id: str,
        parameters_used: Dict[str, Any],
        performance_metrics: Dict[str, float],
        query_type: str,
        query_features: Dict[str, Any]
    ) -> bool:
        """Record feedback about parameter performance.
        
        Args:
            query_id: Unique identifier for the query
            parameters_used: Parameters used for the query
            performance_metrics: Performance metrics from the query
            query_type: Type of query
            query_features: Features of the query
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create feedback entry
            feedback_entry = {
                "query_id": query_id,
                "parameters": parameters_used,
                "metrics": performance_metrics,
                "query_type": query_type,
                "features": query_features,
                "timestamp": time.time()
            }
            
            # Add to history
            self.performance_history.append(feedback_entry)
            
            # Increment pending updates counter for this query type
            self.pending_updates[query_type] += 1
            
            # Save updated history
            self._save_history()
            
            # Train model if we have enough new feedback
            if self.pending_updates[query_type] >= self.min_feedback_samples:
                asyncio.create_task(self.train_model(query_type))
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording feedback: {str(e)}")
            return False
    
    async def train_model(self, query_type: Optional[str] = None) -> Dict[str, Any]:
        """Train parameter model for a specific query type or all types.
        
        Args:
            query_type: Query type to train model for, or None for all types
            
        Returns:
            Information about the trained model(s)
        """
        start_time = time.time()
        model_info = {}
        
        # Determine which query types to train
        query_types_to_train = [query_type] if query_type else list(self.default_parameters.keys())
        
        for qtype in query_types_to_train:
            # Filter history for this query type
            type_history = [
                entry for entry in self.performance_history
                if entry["query_type"] == qtype
            ]
            
            # Skip if not enough data
            if len(type_history) < self.min_feedback_samples:
                logger.info(f"Not enough data to train model for {qtype} ({len(type_history)} samples)")
                model_info[qtype] = {
                    "status": "skipped",
                    "samples": len(type_history)
                }
                continue
            
            try:
                # Extract features and targets
                features = []
                targets = {
                    "alpha": [],
                    "rerank_mmr_lambda": [],
                    "diversity_weight": []
                }
                
                for entry in type_history:
                    # Extract feature vector
                    feature_vector = self._extract_feature_vector(entry["features"])
                    features.append(feature_vector)
                    
                    # Extract target parameters and corresponding performance
                    targets["alpha"].append(
                        (entry["parameters"].get("alpha", 0.5), entry["metrics"].get("overall_score", 0))
                    )
                    targets["rerank_mmr_lambda"].append(
                        (entry["parameters"].get("rerank_mmr_lambda", 0.6), entry["metrics"].get("overall_score", 0))
                    )
                    targets["diversity_weight"].append(
                        (entry["parameters"].get("diversity_weight", 0.2), entry["metrics"].get("overall_score", 0))
                    )
                
                # Create feature scaler
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                # Train models for each parameter
                parameter_models = {}
                
                for param, param_data in targets.items():
                    # Sort by performance
                    param_data.sort(key=lambda x: x[1], reverse=True)
                    
                    # Extract top performing parameter values
                    top_k = max(3, len(param_data) // 3)  # Use top third or at least 3
                    top_values = [val for val, _ in param_data[:top_k]]
                    
                    # Create target array (value of parameter)
                    target_values = [val for val, _ in param_data]
                    
                    # Train Bayesian Ridge model
                    model = BayesianRidge()
                    model.fit(scaled_features, target_values)
                    
                    # Store model
                    parameter_models[param] = model
                
                # Store models and scaler
                self.parameter_models[qtype] = parameter_models
                self.scalers[qtype] = scaler
                
                # Save model
                self._save_model(qtype)
                
                # Update model info
                model_info[qtype] = {
                    "status": "trained",
                    "samples": len(type_history),
                    "training_time": time.time() - start_time
                }
                
                logger.info(f"Trained parameter model for {qtype} with {len(type_history)} samples")
                
            except Exception as e:
                logger.error(f"Error training model for {qtype}: {str(e)}")
                model_info[qtype] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Reset pending updates counter
        for qtype in query_types_to_train:
            self.pending_updates[qtype] = 0
        
        # Update last update time
        self.last_update_time = time.time()
        
        return model_info
    
    def _extract_feature_vector(self, query_features: Dict[str, Any]) -> List[float]:
        """Extract a feature vector from query features.
        
        Args:
            query_features: Dictionary of query features
            
        Returns:
            List of features for model input
        """
        # Define the features we want to extract
        feature_names = [
            "word_count", "sentence_count", "entity_count", 
            "has_numbers", "has_special_characters", "has_question",
            "factual_score", "conceptual_score", "avg_word_length"
        ]
        
        # Create feature vector
        feature_vector = []
        
        for feature in feature_names:
            # Get feature value, default to 0 if not present
            value = query_features.get(feature, 0)
            
            # Convert boolean to int
            if isinstance(value, bool):
                value = int(value)
            
            feature_vector.append(float(value))
        
        return feature_vector
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down Dynamic Parameter Tuner")
        self._save_history()
        for query_type in self.parameter_models:
            self._save_model(query_type)
        
        self.is_running = False
        await self.client.aclose()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the Dynamic Parameter Tuner."""
    from fastapi import FastAPI, HTTPException
    import os
    
    app = FastAPI(title="RAG Dynamic Parameter Tuner", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = DynamicParameterTuner(
        model_dir=os.getenv("PARAMETER_MODEL_DIR", "parameter_models"),
        history_file=os.getenv("PERFORMANCE_HISTORY_FILE", "performance_history.json"),
        cache_service_url=os.getenv("CACHE_SERVICE_URL"),
        min_feedback_samples=int(os.getenv("MIN_FEEDBACK_SAMPLES", "10")),
        update_interval=int(os.getenv("UPDATE_INTERVAL_HOURS", "24"))
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.post("/optimize")
    async def optimize_parameters(request: OptimizeParametersRequest):
        """Optimize parameters for a query."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/feedback")
    async def record_feedback(request: RecordFeedbackRequest):
        """Record feedback about parameter performance."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/train")
    async def train_model(request: TrainModelRequest):
        """Train parameter model."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.get("/shutdown")
    async def shutdown():
        """Shutdown the service."""
        await service.shutdown()
        return {"status": "shutting down"}
    
    return app