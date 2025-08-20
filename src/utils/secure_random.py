"""Secure random utilities for AI Science Platform"""

import secrets
import numpy as np
from typing import Union, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SecureRandomGenerator:
    """Cryptographically secure random number generator for security-sensitive operations"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize secure random generator
        
        Args:
            seed: Optional seed for reproducibility (use only for testing)
        """
        if seed is not None:
            logger.warning("Using seed for SecureRandomGenerator reduces security - use only for testing")
            self._seed = seed
        else:
            self._seed = None
    
    def random_float(self, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Generate cryptographically secure random float
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Secure random float
        """
        if self._seed is not None:
            # For testing with reproducibility
            np.random.seed(self._seed)
            self._seed += 1
            return np.random.uniform(min_val, max_val)
        else:
            # Cryptographically secure
            rand_bytes = secrets.randbelow(2**32)
            normalized = rand_bytes / (2**32 - 1)
            return min_val + normalized * (max_val - min_val)
    
    def random_int(self, min_val: int, max_val: int) -> int:
        """Generate cryptographically secure random integer
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (exclusive)
            
        Returns:
            Secure random integer
        """
        if self._seed is not None:
            # For testing with reproducibility
            np.random.seed(self._seed)
            self._seed += 1
            return np.random.randint(min_val, max_val)
        else:
            # Cryptographically secure
            range_size = max_val - min_val
            return min_val + secrets.randbelow(range_size)
    
    def random_choice(self, choices: List, num_choices: int = 1) -> Union[any, List]:
        """Securely choose random elements from a list
        
        Args:
            choices: List of choices
            num_choices: Number of elements to choose
            
        Returns:
            Single choice or list of choices
        """
        if not choices:
            raise ValueError("Cannot choose from empty list")
        
        if num_choices == 1:
            idx = self.random_int(0, len(choices))
            return choices[idx]
        else:
            selected = []
            available_indices = list(range(len(choices)))
            
            for _ in range(min(num_choices, len(choices))):
                idx = self.random_int(0, len(available_indices))
                chosen_idx = available_indices.pop(idx)
                selected.append(choices[chosen_idx])
            
            return selected
    
    def random_bytes(self, num_bytes: int) -> bytes:
        """Generate cryptographically secure random bytes
        
        Args:
            num_bytes: Number of bytes to generate
            
        Returns:
            Random bytes
        """
        return secrets.token_bytes(num_bytes)
    
    def random_token(self, num_bytes: int = 32) -> str:
        """Generate cryptographically secure random token
        
        Args:
            num_bytes: Number of bytes for token
            
        Returns:
            Random hex token
        """
        return secrets.token_hex(num_bytes)


class ScientificRandomGenerator:
    """Standard random number generator for scientific computing (not security-sensitive)
    
    This class uses numpy's random functions which are appropriate for scientific 
    simulations and algorithms but NOT for security-sensitive operations.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize scientific random generator
        
        Args:
            seed: Optional seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"Scientific random generator initialized with seed {seed}")
        else:
            logger.info("Scientific random generator initialized without seed")
    
    def random_float(self, min_val: float = 0.0, max_val: float = 1.0, 
                    size: Optional[Tuple] = None) -> Union[float, np.ndarray]:
        """Generate random float(s) for scientific computing
        
        Args:
            min_val: Minimum value
            max_val: Maximum value
            size: Shape of output array
            
        Returns:
            Random float or array
        """
        return np.random.uniform(min_val, max_val, size=size)
    
    def random_int(self, min_val: int, max_val: int, 
                  size: Optional[Tuple] = None) -> Union[int, np.ndarray]:
        """Generate random integer(s) for scientific computing
        
        Args:
            min_val: Minimum value (inclusive)
            max_val: Maximum value (exclusive)
            size: Shape of output array
            
        Returns:
            Random integer or array
        """
        return np.random.randint(min_val, max_val, size=size)
    
    def random_choice(self, choices: Union[List, np.ndarray], size: Optional[int] = None,
                     replace: bool = True) -> Union[any, np.ndarray]:
        """Choose random elements for scientific computing
        
        Args:
            choices: Array of choices
            size: Number of elements to choose
            replace: Whether to choose with replacement
            
        Returns:
            Random choice(s)
        """
        return np.random.choice(choices, size=size, replace=replace)
    
    def random_normal(self, mean: float = 0.0, std: float = 1.0,
                     size: Optional[Tuple] = None) -> Union[float, np.ndarray]:
        """Generate random normal distribution values
        
        Args:
            mean: Mean of distribution
            std: Standard deviation
            size: Shape of output array
            
        Returns:
            Random normal values
        """
        return np.random.normal(mean, std, size=size)
    
    def random_array(self, shape: Tuple, distribution: str = 'uniform') -> np.ndarray:
        """Generate random array with specified distribution
        
        Args:
            shape: Shape of array
            distribution: Type of distribution ('uniform', 'normal', 'exponential')
            
        Returns:
            Random array
        """
        if distribution == 'uniform':
            return np.random.random(shape)
        elif distribution == 'normal':
            return np.random.randn(*shape)
        elif distribution == 'exponential':
            return np.random.exponential(size=shape)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")


# Global instances for convenience
secure_random = SecureRandomGenerator()
scientific_random = ScientificRandomGenerator()


def get_secure_random(seed: Optional[int] = None) -> SecureRandomGenerator:
    """Get secure random generator instance
    
    Args:
        seed: Optional seed for testing (reduces security)
        
    Returns:
        SecureRandomGenerator instance
    """
    return SecureRandomGenerator(seed=seed)


def get_scientific_random(seed: Optional[int] = None) -> ScientificRandomGenerator:
    """Get scientific random generator instance
    
    Args:
        seed: Optional seed for reproducibility
        
    Returns:
        ScientificRandomGenerator instance
    """
    return ScientificRandomGenerator(seed=seed)


def replace_insecure_random_usage():
    """Helper function to guide developers on replacing insecure random usage"""
    guidance = """
    RANDOM USAGE SECURITY GUIDANCE:
    
    🔒 SECURITY-SENSITIVE OPERATIONS (use SecureRandomGenerator):
    - Authentication tokens, session IDs
    - Cryptographic keys, salts, nonces  
    - Security-related sampling or selection
    - Random delays for timing attack protection
    - Any operation where predictability = security risk
    
    🔬 SCIENTIFIC COMPUTING (use ScientificRandomGenerator or np.random):
    - Algorithm randomization (e.g., random sampling in ML)
    - Statistical simulations and Monte Carlo methods
    - Random initialization of model parameters  
    - Data augmentation and preprocessing
    - Research experiments requiring reproducibility
    
    Example migration:
    # BEFORE (potentially insecure):
    if np.random.random() < threshold:
        generate_auth_token()
    
    # AFTER (secure):
    from utils.secure_random import secure_random
    if secure_random.random_float() < threshold:
        generate_auth_token()
    
    # SCIENTIFIC (appropriate use of np.random):
    np.random.seed(42)  # For reproducible research
    model_weights = np.random.randn(100, 50)
    """
    
    return guidance