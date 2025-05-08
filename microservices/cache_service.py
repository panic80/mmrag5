"""Cache Service implementation.

This service provides distributed caching capabilities for all other microservices,
enabling performance improvements and reducing duplicate computation.
"""
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx
from pydantic import BaseModel, Field

from microservices.service_interfaces import (
    CacheServiceInterface,
    ServiceRequest,
    ServiceResponse
)

# Configure logging
logger = logging.getLogger(__name__)


class CacheGetRequest(ServiceRequest):
    """Cache get request model."""
    key: str


class CacheSetRequest(ServiceRequest):
    """Cache set request model."""
    key: str
    value: Any
    ttl: Optional[int] = None


class CacheDeleteRequest(ServiceRequest):
    """Cache delete request model."""
    key: str


class CacheResponse(ServiceResponse):
    """Cache response model."""
    key: Optional[str] = None
    success: Optional[bool] = None


class CacheService(CacheServiceInterface):
    """Implementation of the Distributed Cache Service."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_password: Optional[str] = None,
        default_ttl: int = 3600,
        use_memory_cache: bool = False
    ):
        """Initialize the cache service.
        
        Args:
            redis_url: URL for Redis server (e.g., redis://localhost:6379)
            redis_password: Optional Redis password
            default_ttl: Default Time-To-Live in seconds
            use_memory_cache: Whether to use in-memory cache (for testing/development)
        """
        self.redis_url = redis_url
        self.redis_password = redis_password
        self.default_ttl = default_ttl
        self.use_memory_cache = use_memory_cache or not redis_url
        
        # Initialize Redis client if URL is provided
        self.redis_client = None
        if redis_url and not use_memory_cache:
            self._initialize_redis()
        
        # Initialize in-memory cache if needed
        self.memory_cache = {}
        self.memory_cache_expiry = {}
        
        # State tracking
        self.is_running = True
        
        logger.info(f"Cache Service initialized with {'Redis' if redis_url and not use_memory_cache else 'in-memory'} backend")
    
    def _initialize_redis(self):
        """Initialize Redis client."""
        try:
            import redis.asyncio as aioredis
            
            # Parse Redis URL
            if self.redis_url.startswith("redis://"):
                self.redis_client = aioredis.from_url(
                    self.redis_url,
                    password=self.redis_password,
                    decode_responses=True
                )
                logger.info(f"Redis client initialized with URL: {self.redis_url}")
            else:
                logger.error(f"Invalid Redis URL: {self.redis_url}")
                self.use_memory_cache = True
                
        except ImportError:
            logger.error("Redis package not installed, falling back to in-memory cache")
            self.use_memory_cache = True
            
        except Exception as e:
            logger.error(f"Error initializing Redis client: {str(e)}")
            self.use_memory_cache = True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the service is healthy and return status information."""
        status = {
            "status": "healthy",
            "version": "1.0.0",
            "backend": "redis" if not self.use_memory_cache else "memory"
        }
        
        # Check Redis connection if using Redis
        if not self.use_memory_cache and self.redis_client:
            try:
                await self.redis_client.ping()
                status["redis_status"] = "connected"
            except Exception as e:
                status["redis_status"] = f"error: {str(e)}"
                status["status"] = "degraded"
        
        # Add memory cache stats if using in-memory cache
        if self.use_memory_cache:
            status["memory_cache_size"] = len(self.memory_cache)
            status["memory_cache_keys"] = list(self.memory_cache.keys())[:10]  # Show first 10 keys
        
        return status
    
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request and return a response."""
        if isinstance(request, CacheGetRequest):
            try:
                value = await self.get(request.key)
                
                return CacheResponse(
                    request_id=request.request_id,
                    status="success",
                    key=request.key,
                    data=value
                )
                
            except Exception as e:
                logger.error(f"Error getting cache key {request.key}: {str(e)}")
                return CacheResponse(
                    request_id=request.request_id,
                    status="error",
                    key=request.key,
                    message=f"Error getting cache key: {str(e)}"
                )
                
        elif isinstance(request, CacheSetRequest):
            try:
                success = await self.set(
                    key=request.key,
                    value=request.value,
                    ttl=request.ttl
                )
                
                return CacheResponse(
                    request_id=request.request_id,
                    status="success" if success else "error",
                    key=request.key,
                    success=success
                )
                
            except Exception as e:
                logger.error(f"Error setting cache key {request.key}: {str(e)}")
                return CacheResponse(
                    request_id=request.request_id,
                    status="error",
                    key=request.key,
                    message=f"Error setting cache key: {str(e)}",
                    success=False
                )
                
        elif isinstance(request, CacheDeleteRequest):
            try:
                success = await self.delete(request.key)
                
                return CacheResponse(
                    request_id=request.request_id,
                    status="success",
                    key=request.key,
                    success=success
                )
                
            except Exception as e:
                logger.error(f"Error deleting cache key {request.key}: {str(e)}")
                return CacheResponse(
                    request_id=request.request_id,
                    status="error",
                    key=request.key,
                    message=f"Error deleting cache key: {str(e)}",
                    success=False
                )
                
        else:
            return ServiceResponse(
                request_id=getattr(request, "request_id", "unknown"),
                status="error",
                message="Invalid request type"
            )
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # If using Redis
        if not self.use_memory_cache and self.redis_client:
            try:
                result = await self.redis_client.get(key)
                
                if result is not None:
                    # Try to parse JSON
                    try:
                        return json.loads(result)
                    except json.JSONDecodeError:
                        return result
                
                return None
                
            except Exception as e:
                logger.error(f"Redis error getting key {key}: {str(e)}")
                # Fall back to memory cache
                self.use_memory_cache = True
        
        # If using memory cache
        if key in self.memory_cache:
            # Check if key has expired
            if key in self.memory_cache_expiry:
                expiry_time = self.memory_cache_expiry[key]
                if expiry_time < time.time():
                    # Key has expired, delete it
                    del self.memory_cache[key]
                    del self.memory_cache_expiry[key]
                    return None
            
            return self.memory_cache[key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-To-Live in seconds (defaults to self.default_ttl)
            
        Returns:
            True if successful, False otherwise
        """
        # Default TTL if not provided
        if ttl is None:
            ttl = self.default_ttl
        
        # If using Redis
        if not self.use_memory_cache and self.redis_client:
            try:
                # Serialize value to JSON if it's not a string
                if not isinstance(value, (str, bytes)):
                    value = json.dumps(value)
                
                # Set with expiry
                await self.redis_client.set(key, value, ex=ttl)
                return True
                
            except Exception as e:
                logger.error(f"Redis error setting key {key}: {str(e)}")
                # Fall back to memory cache
                self.use_memory_cache = True
        
        # If using memory cache
        try:
            self.memory_cache[key] = value
            
            # Set expiry time
            if ttl > 0:
                self.memory_cache_expiry[key] = time.time() + ttl
            
            return True
            
        except Exception as e:
            logger.error(f"Memory cache error setting key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        # If using Redis
        if not self.use_memory_cache and self.redis_client:
            try:
                result = await self.redis_client.delete(key)
                return result > 0
                
            except Exception as e:
                logger.error(f"Redis error deleting key {key}: {str(e)}")
                # Fall back to memory cache
                self.use_memory_cache = True
        
        # If using memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            if key in self.memory_cache_expiry:
                del self.memory_cache_expiry[key]
            return True
        
        return False
    
    async def flush(self) -> bool:
        """Flush the entire cache.
        
        Returns:
            True if successful, False otherwise
        """
        # If using Redis
        if not self.use_memory_cache and self.redis_client:
            try:
                await self.redis_client.flushdb()
                return True
                
            except Exception as e:
                logger.error(f"Redis error flushing cache: {str(e)}")
                # Fall back to memory cache
                self.use_memory_cache = True
        
        # If using memory cache
        try:
            self.memory_cache.clear()
            self.memory_cache_expiry.clear()
            return True
            
        except Exception as e:
            logger.error(f"Memory cache error flushing cache: {str(e)}")
            return False
    
    async def _cleanup_expired_keys(self):
        """Clean up expired keys from memory cache."""
        if self.use_memory_cache:
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self.memory_cache_expiry.items()
                if expiry < current_time
            ]
            
            for key in expired_keys:
                del self.memory_cache[key]
                del self.memory_cache_expiry[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache keys")
    
    async def _cleanup_loop(self):
        """Background task to periodically clean up expired keys."""
        while self.is_running:
            await self._cleanup_expired_keys()
            await asyncio.sleep(60)  # Clean up every minute
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service."""
        logger.info("Shutting down Cache Service")
        self.is_running = False
        
        # Close Redis connection if using Redis
        if not self.use_memory_cache and self.redis_client:
            await self.redis_client.close()


# FastAPI specific code
def create_fastapi_app():
    """Create a FastAPI app for the Cache Service."""
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    import os
    
    app = FastAPI(title="RAG Cache Service", version="1.0.0")
    
    # Initialize service with environment variables or defaults
    service = CacheService(
        redis_url=os.getenv("REDIS_URL"),
        redis_password=os.getenv("REDIS_PASSWORD"),
        default_ttl=int(os.getenv("DEFAULT_TTL", "3600")),
        use_memory_cache=os.getenv("USE_MEMORY_CACHE", "false").lower() == "true"
    )
    
    @app.on_event("startup")
    async def startup_event():
        # Start cleanup task
        asyncio.create_task(service._cleanup_loop())
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return await service.health_check()
    
    @app.get("/get")
    async def get(key: str):
        """Get a value from the cache."""
        response = await service.process_request(CacheGetRequest(
            request_id=f"get_{int(time.time())}",
            key=key
        ))
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/set")
    async def set(request: CacheSetRequest):
        """Set a value in the cache."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/delete")
    async def delete(request: CacheDeleteRequest):
        """Delete a value from the cache."""
        response = await service.process_request(request)
        if response.status == "error":
            raise HTTPException(status_code=400, detail=response.message)
        return response
    
    @app.post("/flush")
    async def flush():
        """Flush the entire cache."""
        success = await service.flush()
        return {"status": "success" if success else "error"}
    
    @app.get("/shutdown")
    async def shutdown():
        """Shutdown the service."""
        await service.shutdown()
        return {"status": "shutting down"}
    
    return app