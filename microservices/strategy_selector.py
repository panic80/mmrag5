"""Automated Strategy Selector implementation.

This service is responsible for automatically selecting the best retrieval strategy
based on query characteristics, user context, and historical performance data.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from microservices.service_interfaces import (
    ServiceRequest,
    ServiceResponse
)

# Configure logging
logger = logging.getLogger(__name__)


class SelectStrategyRequest(ServiceRequest):
    """Select strategy request model."""
    query_text: str
    user_context: Optional[Dict[str, Any]] = None
    collection: Optional[str] = None


class RecordStrategyFeedbackRequest(ServiceRequest):
    """Record strategy feedback request model."""
    query_id: str
    query_text: str
    strategy_used: Dict[str, Any]
    performance_metrics: Dict[str, float]
    user_context: Optional[Dict[str, Any]] = None


class TrainStrategyModelRequest(ServiceRequest):
    """Train strategy model request."""
    pass


class StrategyResponse(ServiceResponse):
    """Strategy selector response model."""
    strategy: Optional[Dict[str, Any]] = None
    strategy_info: Optional[Dict[str, Any]] = None


class StrategySelector:
    """Automated Strategy Selector for selecting retrieval strategies."""
    
    def __init__(
        self,
        model_dir: str = "strategy_models",
        history_file: str = "strategy_history.json",
        cache_service_url: Optional[str] = None,
        parameter_tuner_url: Optional[str] = None,
        min_feedback_samples: int = 30,
        update_interval: int = 24  # Hours
    ):
        """Initialize the strategy selector.
        
        Args:
            model_dir: Directory to store strategy models
            history_file: File to store strategy performance history
            cache_service_url: Optional URL for the Cache Service
            parameter_tuner_url: Optional URL for the Parameter Tuner Service
            min_feedback_samples: Minimum feedback samples before training
            update_interval: Hours between model updates
        """
        self.model_dir = model_dir
        self.history_file = os.path.join(model_dir, history_file)
        self.cache_service_url = cache_service_url
        self.parameter_tuner_url = parameter_tuner_url
        self.min_feedback_samples = min_feedback_samples
        self.update_interval = update_interval * 3600  # Convert to seconds
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize strategy models
        self.strategy_model = None
        self.strategy_scaler = None
        self.predefined_strategies = self._get_predefined_strategies()
        
        # Performance history
        self.strategy_history = []
        self._load_history()
        
        # Load existing models
        self._load_model()
        
        # Tracking variables
        self.last_update_time = time.time()
        self.pending_updates = 0
        
        # Initialize httpx client
        self.client = httpx.AsyncClient(timeout=30.0)
        self.is_running = True
        
        logger.info(f"Strategy Selector initialized")
    
    def _get_predefined_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined strategies."""
        return {
            "bm25_only": {
                "primary_retriever": "bm25",
                "secondary_retriever": None,
                "fusion_weight": 0.0,
                "fusion_method": None,
                "use_colbert": False,
                "use_splade": False,
                "rerank_method": None
            },
            "vector_only": {
                "primary_retriever": "vector",
                "secondary_retriever": None,
                "fusion_weight": 1.0,
                "fusion_method": None,
                "use_colbert": False,
                "use_splade": False,
                "rerank_method": None
            },
            "hybrid_balanced": {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": 0.5,
                "fusion_method": "linear",
                "use_colbert": False,
                "use_splade": False,
                "rerank_method": "mmr"
            },
            "hybrid_bm25_heavy": {
                "primary_retriever": "bm25",
                "secondary_retriever": "vector",
                "fusion_weight": 0.3,
                "fusion_method": "rrf",
                "use_colbert": False,
                "use_splade": True,
                "rerank_method": "mmr"
            },
            "hybrid_vector_heavy": {
                "primary_retriever": "vector",
                "secondary_retriever": "bm25",
                "fusion_weight": 0.7,
                "fusion_method": "softmax",
                "use_colbert": True,
                "use_splade": False,
                "rerank_method": "context_aware"
            },
            "colbert_enhanced": {
                "primary_retriever": "vector",
                "secondary_retriever": "bm25",
                "fusion_weight": 0.6,
                "fusion_method": "linear",
                "use_colbert": True,
                "use_splade": False,
                "rerank_method": "context_aware"
            },
            "splade_enhanced": {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": 0.4,
                "fusion_method": "rrf",
                "use_colbert": False,
                "use_splade": True,
                "rerank_method": "mmr"
            },
            "advanced_fusion": {
                "primary_retriever": "hybrid",
                "secondary_retriever": None,
                "fusion_weight": 0.5,
                "fusion_method": "harmonic",
                "use_colbert": True,
                "use_splade": True,
                "rerank_method": "diversity"
            }
        }
    
    def _load_history(self):
        """Load strategy performance history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.strategy_history = json.load(f)
                logger.info(f"Loaded {len(self.strategy_history)} strategy history entries")
            except Exception as e:
                logger.error(f"Error loading strategy history: {str(e)}")
                self.strategy_history = []
    
    def _save_history(self):
        """Save strategy performance history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.strategy_history, f)
            logger.info(f"Saved {len(self.strategy_history)} strategy history entries")
        except Exception as e:
            logger.error(f"Error saving strategy history: {str(e)}")
    
    def _load_model(self):
        """Load strategy selection model from disk."""
        model_path = os.path.join(self.model_dir, "strategy_model.pkl")
        scaler_path = os.path.join(self.model_dir, "strategy_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                with open(model_path, 'rb') as f:
                    self.strategy_model = pickle.load(f)
                
                with open(scaler_path, 'rb') as f:
                    self.strategy_scaler = pickle.load(f)
                
                logger.info(f"Loaded strategy selection model")
            except Exception as e:
                logger.error(f"Error loading strategy model: {str(e)}")
    
    def _save_model(self):
        """Save strategy selection model to disk."""
        if self.strategy_model and self.strategy_scaler:
            model_path = os.path.join(self.model_dir, "strategy_model.pkl")
            scaler_path = os.path.join(self.model_dir, "strategy_scaler.pkl")
            
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(self.strategy_model, f)
                
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.strategy_scaler, f)
                
                logger.info(f"Saved strategy selection model")
            except Exception as e:
                logger.error(f"Error saving strategy model: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        status = {
            "status": "healthy",
            "version": "1.0.0",
            "model_loaded": self.strategy_model is not None,
            "feedback_entries": len(self.strategy_history),
            "last_update_time": self.last_update_time,
            "predefined_strategies": list(self.predefined_strategies.keys())
        }
        
        # Check cache service if configured
        if self.cache_service_url:
            try:
                response = await self.client.get(f"{self.cache_service_url}/health")
                status["cache_status"] = "connected" if response.status_code == 200 else "error"
            except Exception as e:
                status["cache_status"] = f"error: {str(e)}"
                
        # Check parameter tuner service if configured
        if self.parameter_tuner_url:
            try:
                response = await self.client.get(f"{self.parameter_tuner_url}/health")
                status["parameter_tuner_status"] = "connected" if response.status_code == 200 else "error"
            except Exception as e:
                status["parameter_tuner_status"] = f"error: {str(e)}"
        
        return status
    
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request and return a response."""
        if isinstance(request, SelectStrategyRequest):
            try:
                strategy, strategy_info = await self.select_strategy(
                    query_text=request.query_text,
                    user_context=request.user_context,
                    collection=request.collection
                )
                
                return StrategyResponse(
                    request_id=request.request_id,
                    status="success",
                    data={
                        "strategy": strategy,
                        "strategy_info": strategy_info
                    }
                )
                
            except Exception as e:
                logger.error(f"Error selecting strategy: {str(e)}")
                return StrategyResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error selecting strategy: {str(e)}"
                )
                
        elif isinstance(request, RecordStrategyFeedbackRequest):
            try:
                await self.record_feedback(
                    query_id=request.query_id,
                    query_text=request.query_text,
                    strategy_used=request.strategy_used,
                    performance_metrics=request.performance_metrics,
                    user_context=request.user_context
                )
                
                return ServiceResponse(
                    request_id=request.request_id,
                    status="success",
                    message="Strategy feedback recorded successfully"
                )
                
            except Exception as e:
                logger.error(f"Error recording strategy feedback: {str(e)}")
                return ServiceResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error recording strategy feedback: {str(e)}"
                )
                
        elif isinstance(request, TrainStrategyModelRequest):
            try:
                model_info = await self.train_model()
                
                return StrategyResponse(
                    request_id=request.request_id,
                    status="success",
                    data={"strategy_info": model_info}
                )
                
            except Exception as e:
                logger.error(f"Error training strategy model: {str(e)}")
                return StrategyResponse(
                    request_id=request.request_id,
                    status="error",
                    message=f"Error training strategy model: {str(e)}"
                )
                
        else:
            return ServiceResponse(
                request_id=getattr(request, "request_id", "unknown"),
                status="error",
                message="Invalid request type"
            )
    
    async def select_strategy(
        self, 
        query_text: str, 
        user_context: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select the best retrieval strategy for a query.
        
        Args:
            query_text: The query text
            user_context: Optional user context information
            collection: Optional collection name
            
        Returns:
            Tuple of (strategy, strategy_info)
        """
        # Check cache first if cache service is configured
        if self.cache_service_url:
            import hashlib
            
            # Generate a deterministic representation of inputs for caching
            cache_input = {
                "query_text": query_text,
                "user_context": user_context or {},
                "collection": collection
            }
            
            cache_str = json.dumps(cache_input, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
            cache_key = f"strategy:{cache_hash}"
            
            try:
                cache_response = await self.client.get(
                    f"{self.cache_service_url}/get",
                    params={"key": cache_key}
                )
                
                if cache_response.status_code == 200:
                    cached_data = cache_response.json().get("data")
                    if cached_data:
                        logger.info(f"Cache hit for strategy selection")
                        return cached_data["strategy"], cached_data["strategy_info"]
            except Exception as e:
                logger.warning(f"Error checking cache: {str(e)}")
        
        # Check if we need to update the model
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            # Check if we have enough new feedback to warrant an update
            if self.pending_updates >= self.min_feedback_samples:
                await self.train_model()
            
            self.last_update_time = current_time
            self.pending_updates = 0
        
        start_time = time.time()
        
        # Extract query features
        query_features = await self._extract_query_features(query_text)
        
        # Determine query type
        query_type = self._determine_query_type(query_features)
        
        # Strategy selection approach depends on whether we have a trained model
        strategy_info = {
            "query_type": query_type,
            "selection_method": "model" if self.strategy_model else "rule_based",
            "query_features": query_features
        }
        
        if self.strategy_model and self.strategy_scaler:
            # Use ML model for selection
            try:
                # Extract feature vector
                feature_vector = self._extract_feature_vector(query_features, user_context)
                
                # Scale features
                scaled_features = self.strategy_scaler.transform([feature_vector])
                
                # Predict strategy
                strategy_probs = self.strategy_model.predict_proba(scaled_features)[0]
                strategy_names = self.strategy_model.classes_
                
                # Get top 3 strategies with probabilities
                top_indices = np.argsort(strategy_probs)[::-1][:3]
                top_strategies = [
                    (strategy_names[i], float(strategy_probs[i])) 
                    for i in top_indices
                ]
                
                # Use the highest probability strategy
                selected_strategy_name = top_strategies[0][0]
                selected_strategy = self.predefined_strategies[selected_strategy_name].copy()
                
                # Add strategy selection info
                strategy_info["model_confidence"] = top_strategies[0][1]
                strategy_info["top_strategies"] = top_strategies
                strategy_info["selection_time"] = time.time() - start_time
                
            except Exception as e:
                logger.error(f"Error using strategy model: {str(e)}")
                # Fall back to rule-based selection
                selected_strategy, rule_info = self._rule_based_selection(query_type, query_features, user_context)
                strategy_info.update(rule_info)
                strategy_info["selection_method"] = "rule_based_fallback"
                strategy_info["model_error"] = str(e)
        else:
            # Use rule-based selection
            selected_strategy, rule_info = self._rule_based_selection(query_type, query_features, user_context)
            strategy_info.update(rule_info)
        
        # If parameter tuner is available, optimize parameters
        if self.parameter_tuner_url:
            try:
                # Request optimized parameters
                response = await self.client.post(
                    f"{self.parameter_tuner_url}/optimize",
                    json={
                        "request_id": f"strategy_{int(time.time())}",
                        "query_type": query_type,
                        "query_features": query_features,
                        "query_text": query_text
                    }
                )
                
                if response.status_code == 200:
                    # Get optimized parameters
                    optimized_params = response.json().get("data", {}).get("optimized_parameters", {})
                    
                    # Update strategy with optimized parameters
                    if optimized_params:
                        # Copy relevant parameters
                        if "alpha" in optimized_params:
                            selected_strategy["fusion_weight"] = optimized_params["alpha"]
                        
                        if "use_colbert" in optimized_params:
                            selected_strategy["use_colbert"] = optimized_params["use_colbert"]
                            
                        if "use_splade" in optimized_params:
                            selected_strategy["use_splade"] = optimized_params["use_splade"]
                            
                        if "rerank_mmr_lambda" in optimized_params:
                            selected_strategy["rerank_mmr_lambda"] = optimized_params["rerank_mmr_lambda"]
                        
                        strategy_info["parameters_tuned"] = True
                        
            except Exception as e:
                logger.warning(f"Error tuning parameters: {str(e)}")
                strategy_info["parameters_tuned"] = False
                strategy_info["tuning_error"] = str(e)
        
        # Cache result if cache service is configured
        if self.cache_service_url:
            import hashlib
            
            cache_input = {
                "query_text": query_text,
                "user_context": user_context or {},
                "collection": collection
            }
            
            cache_str = json.dumps(cache_input, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
            cache_key = f"strategy:{cache_hash}"
            
            try:
                await self.client.post(
                    f"{self.cache_service_url}/set",
                    json={
                        "key": cache_key,
                        "value": {
                            "strategy": selected_strategy,
                            "strategy_info": strategy_info
                        },
                        "ttl": 3600  # 1 hour TTL
                    }
                )
            except Exception as e:
                logger.warning(f"Error caching strategy: {str(e)}")
        
        return selected_strategy, strategy_info
    
    def _rule_based_selection(
        self, 
        query_type: str, 
        query_features: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply rule-based strategy selection.
        
        Args:
            query_type: The determined query type
            query_features: Features extracted from the query
            user_context: Optional user context information
            
        Returns:
            Tuple of (strategy, strategy_info)
        """
        # Start with a default strategy based on query type
        if query_type == "factual":
            strategy_name = "hybrid_bm25_heavy"
        elif query_type == "conceptual":
            strategy_name = "hybrid_vector_heavy"
        else:  # balanced
            strategy_name = "hybrid_balanced"
        
        # Apply specialized rules
        if query_features.get("entity_count", 0) > 2:
            # Queries with many entities benefit from SPLADE
            strategy_name = "splade_enhanced"
        
        if query_features.get("is_complex", False) or query_features.get("word_count", 0) > 15:
            # Complex or long queries benefit from ColBERT
            strategy_name = "colbert_enhanced"
        
        if query_features.get("diversity_needed", False) or query_features.get("is_broad", False):
            # Queries needing diverse results benefit from advanced fusion
            strategy_name = "advanced_fusion"
        
        # Apply user context rules if available
        if user_context:
            # Check if user has preferred mode
            if user_context.get("preferred_mode") == "precise":
                strategy_name = "hybrid_bm25_heavy"
            elif user_context.get("preferred_mode") == "recall":
                strategy_name = "hybrid_vector_heavy"
            
            # Check if search history suggests a preference
            if user_context.get("history_preference") == "factual":
                strategy_name = "hybrid_bm25_heavy"
            elif user_context.get("history_preference") == "conceptual":
                strategy_name = "colbert_enhanced"
        
        # Get the strategy
        strategy = self.predefined_strategies[strategy_name].copy()
        
        # Return strategy and selection info
        rule_info = {
            "rule_based_strategy": strategy_name,
            "rule_factors": {
                "entity_rich": query_features.get("entity_count", 0) > 2,
                "complex_query": query_features.get("is_complex", False) or query_features.get("word_count", 0) > 15,
                "diversity_needed": query_features.get("diversity_needed", False) or query_features.get("is_broad", False)
            }
        }
        
        return strategy, rule_info
    
    async def _extract_query_features(self, query_text: str) -> Dict[str, Any]:
        """Extract features from a query text.
        
        Args:
            query_text: The query text
            
        Returns:
            Dictionary of query features
        """
        import re
        
        # Basic statistics
        word_count = len(query_text.split())
        sentence_count = len(re.split(r'[.!?]+', query_text))
        avg_word_length = sum(len(word) for word in query_text.split()) / max(1, word_count)
        
        # Check for special characters
        has_special_chars = bool(re.search(r'[^a-zA-Z0-9\s]', query_text))
        has_numbers = bool(re.search(r'\d+', query_text))
        
        # Check for question
        is_question = query_text.strip().endswith('?') or query_text.lower().startswith(('who', 'what', 'when', 'where', 'why', 'how'))
        
        # Extract entities (simple approach - look for capitalized words)
        entities = re.findall(r'\b[A-Z][a-zA-Z]*\b', query_text)
        entity_count = len(entities)
        
        # Check complexity and broadness
        is_complex = word_count > 10 or sentence_count > 1
        is_broad = not entity_count and not is_question and word_count < 5
        
        # Check query intent
        # Factual indicators
        factual_indicators = ['who', 'what', 'when', 'where', 'which', 'how many', 'list', 'name']
        factual_score = sum(1 for word in factual_indicators if word.lower() in query_text.lower().split())
        
        # Conceptual indicators
        conceptual_indicators = ['why', 'how', 'explain', 'describe', 'compare', 'analyze', 'evaluate']
        conceptual_score = sum(1 for word in conceptual_indicators if word.lower() in query_text.lower().split())
        
        # Check for diversity need
        diversity_needed = "vs" in query_text.lower().split() or "versus" in query_text.lower().split() or "compare" in query_text.lower()
        
        # Assemble features
        features = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "has_special_characters": has_special_chars,
            "has_numbers": has_numbers,
            "is_question": is_question,
            "entity_count": entity_count,
            "is_complex": is_complex,
            "is_broad": is_broad,
            "factual_score": factual_score,
            "conceptual_score": conceptual_score,
            "diversity_needed": diversity_needed
        }
        
        return features
    
    def _determine_query_type(self, query_features: Dict[str, Any]) -> str:
        """Determine the query type based on features.
        
        Args:
            query_features: Features of the query
            
        Returns:
            Query type (factual, conceptual, or balanced)
        """
        factual_score = query_features.get("factual_score", 0)
        conceptual_score = query_features.get("conceptual_score", 0)
        
        if factual_score > conceptual_score:
            return "factual"
        elif conceptual_score > factual_score:
            return "conceptual"
        else:
            return "balanced"
    
    def _extract_feature_vector(
        self, 
        query_features: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Extract a feature vector for model input.
        
        Args:
            query_features: Features extracted from the query
            user_context: Optional user context information
            
        Returns:
            Feature vector for model input
        """
        # Define the features we want to use from query_features
        query_feature_names = [
            "word_count", "sentence_count", "avg_word_length",
            "has_special_characters", "has_numbers", "is_question",
            "entity_count", "is_complex", "is_broad",
            "factual_score", "conceptual_score", "diversity_needed"
        ]
        
        # Create feature vector from query features
        feature_vector = []
        
        for feature in query_feature_names:
            # Get feature value, default to 0 if not present
            value = query_features.get(feature, 0)
            
            # Convert boolean to int
            if isinstance(value, bool):
                value = int(value)
            
            feature_vector.append(float(value))
        
        # Add user context features if available
        if user_context:
            # Preference for precise results (0-1)
            precision_pref = float(user_context.get("precision_preference", 0.5))
            feature_vector.append(precision_pref)
            
            # Preference for diverse results (0-1)
            diversity_pref = float(user_context.get("diversity_preference", 0.5))
            feature_vector.append(diversity_pref)
            
            # User expertise level (0-1)
            expertise = float(user_context.get("expertise_level", 0.5))
            feature_vector.append(expertise)
        else:
            # Default values if no user context
            feature_vector.extend([0.5, 0.5, 0.5])
        
        return feature_vector
    
    async def record_feedback(
        self,
        query_id: str,
        query_text: str,
        strategy_used: Dict[str, Any],
        performance_metrics: Dict[str, float],
        user_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record feedback about strategy performance.
        
        Args:
            query_id: Unique identifier for the query
            query_text: The query text
            strategy_used: Strategy used for the query
            performance_metrics: Performance metrics from the query
            user_context: Optional user context information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract query features
            query_features = await self._extract_query_features(query_text)
            
            # Determine query type
            query_type = self._determine_query_type(query_features)
            
            # Determine strategy name
            strategy_name = "custom"
            for name, strategy in self.predefined_strategies.items():
                if all(strategy.get(key) == strategy_used.get(key) for key in strategy):
                    strategy_name = name
                    break
            
            # Create feedback entry
            feedback_entry = {
                "query_id": query_id,
                "query_text": query_text,
                "query_type": query_type,
                "query_features": query_features,
                "strategy_name": strategy_name,
                "strategy": strategy_used,
                "metrics": performance_metrics,
                "user_context": user_context,
                "timestamp": time.time()
            }
            
            # Add to history
            self.strategy_history.append(feedback_entry)
            
            # Increment pending updates counter
            self.pending_updates += 1
            
            # Save updated history
            self._save_history()
            
            # Train model if we have enough new feedback
            if self.pending_updates >= self.min_feedback_samples:
                asyncio.create_task(self.train_model())
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording strategy feedback: {str(e)}")
            return False
    
    async def train_model(self) -> Dict[str, Any]:
        """Train strategy selection model.
        
        Returns:
            Information about the trained model
        """
        start_time = time.time()
        
        # Check if we have enough data
        if len(self.strategy_history) < self.min_feedback_samples:
            logger.info(f"Not enough data to train model ({len(self.strategy_history)} samples)")
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "samples": len(self.strategy_history),
                "min_required": self.min_feedback_samples
            }
        
        try:
            # Extract features and targets
            features = []
            targets = []
            
            for entry in self.strategy_history:
                # Get query features
                query_features = entry["query_features"]
                
                # Get user context if available
                user_context = entry.get("user_context")
                
                # Extract feature vector
                feature_vector = self._extract_feature_vector(query_features, user_context)
                
                # Get strategy name as target
                strategy_name = entry["strategy_name"]
                
                # Only use predefined strategies as targets
                if strategy_name in self.predefined_strategies:
                    features.append(feature_vector)
                    targets.append(strategy_name)
            
            # Check if we have enough samples after filtering
            if len(features) < self.min_feedback_samples:
                logger.info(f"Not enough valid strategy samples ({len(features)} samples)")
                return {
                    "status": "skipped",
                    "reason": "insufficient_valid_data",
                    "valid_samples": len(features),
                    "min_required": self.min_feedback_samples
                }
            
            # Create feature scaler
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Train RandomForest classifier
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(scaled_features, targets)
            
            # Store model and scaler
            self.strategy_model = model
            self.strategy_scaler = scaler
            
            # Save model
            self._save_model()
            
            # Calculate training statistics
            feature_importances = model.feature_importances_
            
            # Class distribution
            class_distribution = {
                cls: int(sum(1 for t in targets if t == cls))
                for cls in model.classes_
            }
            
            # Reset pending updates counter
            self.pending_updates = 0
            
            # Update last update time
            self.last_update_time = time.time()
            
            # Return model info
            model_info = {
                "status": "trained",
                "samples": len(features),
                "training_time": time.time() - start_time,
                "strategies": list(model.classes_),
                "class_distribution": class_distribution,
                "feature_importances": feature_importances.tolist()
            }
            
            logger.info(f"Trained strategy selection model with {len(features)} samples")
            return model_info
            
        except Exception as e:
            logger.error(f"Error training strategy model: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down Strategy Selector")
        self._save_history()
        if self.strategy_model:
            self._save_model()
        
        self.is_running = False
        await self.client.aclose()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the Strategy Selector."""
    from fastapi import FastAPI, HTTPException
    import os
    
    app = FastAPI(title="RAG Strategy Selector", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = StrategySelector(
        model_dir=os.getenv("STRATEGY_MODEL_DIR", "strategy_models"),
        history_file=os.getenv("STRATEGY_HISTORY_FILE", "strategy_history.json"),
        cache_service_url=os.getenv("CACHE_SERVICE_URL"),
        parameter_tuner_url=os.getenv("PARAMETER_TUNER_URL"),
        min_feedback_samples=int(os.getenv("MIN_FEEDBACK_SAMPLES", "30")),
        update_interval=int(os.getenv("UPDATE_INTERVAL_HOURS", "24"))
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.post("/select")
    async def select_strategy(request: SelectStrategyRequest):
        """Select strategy for a query."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/feedback")
    async def record_feedback(request: RecordStrategyFeedbackRequest):
        """Record feedback about strategy performance."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/train")
    async def train_model(request: TrainStrategyModelRequest):
        """Train strategy selection model."""
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