"""Intelligent caching system for performance optimization"""

import time
import hashlib
import pickle
import logging
from typing import Any, Dict, Optional, Callable, Tuple
from functools import wraps
from pathlib import Path
import threading
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache with size and TTL limits"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """Get item from cache"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return False, None
            
            # Check TTL
            if time.time() - self._timestamps[key] > self.ttl:
                self._remove(key)
                self._misses += 1
                return False, None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return True, self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                # Check size limit
                if len(self._cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key = next(iter(self._cache))
                    self._remove(oldest_key)
                
                self._cache[key] = value
            
            self._timestamps[key] = time.time()
    
    def _remove(self, key: str) -> None:
        """Remove item from cache"""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl": self.ttl
            }


class PersistentCache:
    """Disk-based persistent cache with compression"""
    
    def __init__(self, cache_dir: str = "./.cache", max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self._lock = threading.Lock()
        self._index_file = self.cache_dir / "cache_index.pkl"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk"""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk"""
        try:
            with open(self._index_file, 'wb') as f:
                pickle.dump(self._index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key"""
        # Use hash to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """Get item from persistent cache"""
        with self._lock:
            if key not in self._index:
                return False, None
            
            file_path = self._get_file_path(key)
            if not file_path.exists():
                # Remove stale index entry
                del self._index[key]
                self._save_index()
                return False, None
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access time
                self._index[key]["last_access"] = time.time()
                self._save_index()
                
                return True, data
            
            except Exception as e:
                logger.error(f"Failed to load cached data: {e}")
                return False, None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in persistent cache"""
        with self._lock:
            try:
                file_path = self._get_file_path(key)
                
                # Save data to disk
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update index
                file_size = file_path.stat().st_size
                self._index[key] = {
                    "file_path": str(file_path),
                    "size_bytes": file_size,
                    "created": time.time(),
                    "last_access": time.time()
                }
                
                # Clean up if over size limit
                self._cleanup_if_needed()
                self._save_index()
                
            except Exception as e:
                logger.error(f"Failed to cache data: {e}")
    
    def _cleanup_if_needed(self) -> None:
        """Remove old cache entries if size limit exceeded"""
        total_size_mb = sum(entry["size_bytes"] for entry in self._index.values()) / (1024 * 1024)
        
        if total_size_mb > self.max_size_mb:
            # Sort by last access time (oldest first)
            sorted_entries = sorted(
                self._index.items(), 
                key=lambda x: x[1]["last_access"]
            )
            
            # Remove oldest entries until under limit
            for key, entry in sorted_entries:
                if total_size_mb <= self.max_size_mb * 0.8:  # Leave some buffer
                    break
                
                try:
                    Path(entry["file_path"]).unlink(missing_ok=True)
                    del self._index[key]
                    total_size_mb -= entry["size_bytes"] / (1024 * 1024)
                    logger.info(f"Removed cache entry: {key}")
                except Exception as e:
                    logger.error(f"Failed to remove cache entry {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            for entry in self._index.values():
                try:
                    Path(entry["file_path"]).unlink(missing_ok=True)
                except Exception:
                    pass
            
            self._index.clear()
            self._save_index()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size_mb = sum(entry["size_bytes"] for entry in self._index.values()) / (1024 * 1024)
            
            return {
                "entries": len(self._index),
                "total_size_mb": total_size_mb,
                "max_size_mb": self.max_size_mb,
                "cache_dir": str(self.cache_dir)
            }


class CacheManager:
    """Centralized cache management with multiple cache levels"""
    
    def __init__(self, 
                 memory_cache_size: int = 1000,
                 memory_ttl: int = 3600,
                 disk_cache_size_mb: int = 500,
                 cache_dir: str = "./.cache"):
        
        self.memory_cache = LRUCache(memory_cache_size, memory_ttl)
        self.disk_cache = PersistentCache(cache_dir, disk_cache_size_mb)
        self._enabled = True
        logger.info(f"Cache manager initialized: memory={memory_cache_size}, disk={disk_cache_size_mb}MB")
    
    def _generate_key(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate cache key from function name and arguments"""
        try:
            # Handle numpy arrays specially
            processed_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    processed_args.append(f"array_shape_{arg.shape}_hash_{hash(arg.tobytes())}")
                else:
                    processed_args.append(str(arg))
            
            processed_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    processed_kwargs[k] = f"array_shape_{v.shape}_hash_{hash(v.tobytes())}"
                else:
                    processed_kwargs[k] = str(v)
            
            key_data = f"{func_name}:{str(processed_args)}:{str(sorted(processed_kwargs.items()))}"
            return hashlib.sha256(key_data.encode()).hexdigest()
        
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            # Fallback to simple hash
            return hashlib.sha256(str((func_name, args, kwargs)).encode()).hexdigest()
    
    def get(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> Tuple[bool, Any]:
        """Get cached result"""
        if not self._enabled:
            return False, None
        
        key = self._generate_key(func_name, args, kwargs)
        
        # Try memory cache first
        found, value = self.memory_cache.get(key)
        if found:
            logger.debug(f"Memory cache hit for {func_name}")
            return True, value
        
        # Try disk cache
        found, value = self.disk_cache.get(key)
        if found:
            # Promote to memory cache
            self.memory_cache.put(key, value)
            logger.debug(f"Disk cache hit for {func_name}")
            return True, value
        
        logger.debug(f"Cache miss for {func_name}")
        return False, None
    
    def put(self, func_name: str, args: Tuple, kwargs: Dict[str, Any], result: Any) -> None:
        """Store result in cache"""
        if not self._enabled:
            return
        
        key = self._generate_key(func_name, args, kwargs)
        
        # Store in both caches
        self.memory_cache.put(key, result)
        self.disk_cache.put(key, result)
        
        logger.debug(f"Cached result for {func_name}")
    
    def clear(self) -> None:
        """Clear all caches"""
        self.memory_cache.clear()
        self.disk_cache.clear()
        logger.info("All caches cleared")
    
    def enable(self) -> None:
        """Enable caching"""
        self._enabled = True
        logger.info("Caching enabled")
    
    def disable(self) -> None:
        """Disable caching"""
        self._enabled = False
        logger.info("Caching disabled")
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "enabled": self._enabled,
            "memory_cache": self.memory_cache.stats(),
            "disk_cache": self.disk_cache.stats()
        }


# Global cache manager
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached_function(cache_disabled: bool = False, 
                   ttl: Optional[int] = None,
                   memory_only: bool = False):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if cache_disabled:
                return func(*args, **kwargs)
            
            cache_manager = get_cache_manager()
            
            # Check cache first
            found, result = cache_manager.get(func.__name__, args, kwargs)
            if found:
                return result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Only cache if execution took significant time
            if execution_time > 0.1 or not memory_only:
                cache_manager.put(func.__name__, args, kwargs, result)
            
            return result
        
        # Add cache control methods
        wrapper.clear_cache = lambda: get_cache_manager().clear()
        wrapper.cache_stats = lambda: get_cache_manager().stats()
        
        return wrapper
    return decorator


def clear_all_caches() -> None:
    """Clear all global caches"""
    cache_manager = get_cache_manager()
    cache_manager.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    cache_manager = get_cache_manager()
    return cache_manager.stats()