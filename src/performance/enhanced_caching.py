"""Enhanced caching system with intelligent cache management"""

import time
import threading
import logging
from typing import Any, Dict, Optional, Callable, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
import hashlib
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class IntelligentCache:
    """Intelligent caching with adaptive eviction policies"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512, 
                 default_ttl: int = 3600):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"IntelligentCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def _compute_key(self, *args, **kwargs) -> str:
        """Compute cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode()) if hasattr(value, '__str__') else 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                self.stats['misses'] += 1
                return default
            
            # Check TTL
            if time.time() - entry.created_at > self.default_ttl:
                del self.cache[key]
                self.stats['misses'] += 1
                return default
            
            # Update access info
            entry.accessed_at = time.time()
            entry.access_count += 1
            self.stats['hits'] += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, 
            tags: Optional[List[str]] = None) -> bool:
        """Put value in cache"""
        with self.lock:
            size_bytes = self._get_size(value)
            
            # Check if we can fit this entry
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Entry too large for cache: {size_bytes} bytes")
                return False
            
            now = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                accessed_at=now,
                size_bytes=size_bytes,
                tags=tags or []
            )
            
            # Make room if necessary
            self._ensure_space(size_bytes)
            
            self.cache[key] = entry
            self.stats['memory_usage'] += size_bytes
            
            logger.debug(f"Cached entry: {key} ({size_bytes} bytes)")
            return True
    
    def _ensure_space(self, needed_bytes: int):
        """Ensure there's space for new entry"""
        while (len(self.cache) >= self.max_size or 
               self.stats['memory_usage'] + needed_bytes > self.max_memory_bytes):
            
            if not self.cache:
                break
            
            # Find least valuable entry (LRU with access count weighting)
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].accessed_at / max(1, self.cache[k].access_count))
            
            removed_entry = self.cache.pop(oldest_key)
            self.stats['memory_usage'] -= removed_entry.size_bytes
            self.stats['evictions'] += 1
            
            logger.debug(f"Evicted entry: {oldest_key}")
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.stats['memory_usage'] -= entry.size_bytes
                return True
            return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Remove all entries with specific tag"""
        removed = 0
        with self.lock:
            keys_to_remove = [key for key, entry in self.cache.items() 
                            if tag in entry.tags]
            
            for key in keys_to_remove:
                entry = self.cache.pop(key)
                self.stats['memory_usage'] -= entry.size_bytes
                removed += 1
        
        logger.info(f"Invalidated {removed} entries with tag: {tag}")
        return removed
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats['memory_usage'] = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
            
            return {
                'entries': len(self.cache),
                'memory_usage_mb': self.stats['memory_usage'] / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'utilization': len(self.cache) / self.max_size
            }
    
    def _cleanup_worker(self):
        """Background thread for cache maintenance"""
        while True:
            try:
                time.sleep(60)  # Run every minute
                with self.lock:
                    now = time.time()
                    expired_keys = [
                        key for key, entry in self.cache.items()
                        if now - entry.created_at > self.default_ttl
                    ]
                    
                    for key in expired_keys:
                        entry = self.cache.pop(key)
                        self.stats['memory_usage'] -= entry.size_bytes
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                        
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")


class CacheDecorator:
    """Decorator for automatic function result caching"""
    
    def __init__(self, cache: IntelligentCache, ttl: Optional[int] = None,
                 tags: Optional[List[str]] = None, key_func: Optional[Callable] = None):
        self.cache = cache
        self.ttl = ttl
        self.tags = tags or []
        self.key_func = key_func
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            if self.key_func:
                cache_key = self.key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{self.cache._compute_key(*args, **kwargs)}"
            
            # Check cache
            result = self.cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            self.cache.put(cache_key, result, self.ttl, self.tags + [func.__name__])
            
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


# Global cache instance
_global_cache = None

def get_cache() -> IntelligentCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache


def cached(ttl: Optional[int] = None, tags: Optional[List[str]] = None,
          key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    return CacheDecorator(get_cache(), ttl, tags, key_func)


# Example usage
if __name__ == "__main__":
    cache = IntelligentCache(max_size=100, max_memory_mb=50)
    
    @cached(ttl=300, tags=['computation'])
    def expensive_computation(x, y):
        """Simulate expensive computation"""
        time.sleep(0.1)
        return x ** 2 + y ** 2
    
    # Test caching
    start = time.time()
    result1 = expensive_computation(10, 20)
    first_call_time = time.time() - start
    
    start = time.time()
    result2 = expensive_computation(10, 20)
    cached_call_time = time.time() - start
    
    print(f"First call: {first_call_time:.3f}s, Cached call: {cached_call_time:.6f}s")
    print(f"Speedup: {first_call_time / cached_call_time:.1f}x")
    print(f"Cache stats: {cache.get_stats()}")