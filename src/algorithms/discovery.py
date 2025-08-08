"""Core discovery engine for scientific automation"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Discovery:
    """Represents a scientific discovery"""
    hypothesis: str
    evidence: List[Dict[str, Any]]
    confidence: float
    metrics: Dict[str, float]
    timestamp: str
    

class DiscoveryEngine:
    """AI-driven scientific discovery automation engine"""
    
    def __init__(self, discovery_threshold: float = 0.7):
        self.discovery_threshold = discovery_threshold
        self.discoveries = []
        self.hypotheses_tested = 0
        logger.info("DiscoveryEngine initialized")
    
    def generate_hypothesis(self, data: np.ndarray, context: str = "") -> str:
        """Generate scientific hypothesis from data patterns"""
        self.hypotheses_tested += 1
        
        # Simple pattern detection for demonstration
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val < 0.1 * abs(mean_val):
            hypothesis = f"Data shows consistent behavior around {mean_val:.3f} with low variance"
        elif np.any(data > mean_val + 2 * std_val):
            hypothesis = f"Data contains outliers suggesting anomalous behavior above {mean_val + 2 * std_val:.3f}"
        else:
            hypothesis = f"Data exhibits normal distribution pattern with mean {mean_val:.3f}"
        
        if context:
            hypothesis = f"In context '{context}': {hypothesis}"
        
        logger.info(f"Generated hypothesis #{self.hypotheses_tested}: {hypothesis}")
        return hypothesis
    
    def test_hypothesis(self, hypothesis: str, data: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[bool, Dict[str, float]]:
        """Test a scientific hypothesis against data"""
        metrics = {}
        
        # Basic statistical tests
        metrics['data_size'] = len(data)
        metrics['mean'] = float(np.mean(data))
        metrics['std'] = float(np.std(data))
        metrics['skewness'] = float(self._calculate_skewness(data))
        
        if targets is not None:
            correlation = np.corrcoef(data.flatten(), targets.flatten())[0, 1]
            metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
        
        # Simple validation logic
        is_valid = (
            metrics['std'] > 0 and 
            metrics['data_size'] > 10 and
            abs(metrics.get('correlation', 0.5)) > 0.3
        )
        
        logger.info(f"Hypothesis test result: {is_valid}, metrics: {metrics}")
        return is_valid, metrics
    
    def discover(self, data: np.ndarray, targets: Optional[np.ndarray] = None, context: str = "") -> List[Discovery]:
        """Execute full discovery pipeline"""
        logger.info(f"Starting discovery process on data shape {data.shape}")
        
        discoveries = []
        
        # Generate and test multiple hypotheses
        for i in range(3):  # Test 3 different hypotheses
            hypothesis = self.generate_hypothesis(data, f"{context}_variant_{i}")
            is_valid, metrics = self.test_hypothesis(hypothesis, data, targets)
            
            if is_valid:
                confidence = min(0.95, 0.5 + metrics.get('correlation', 0.0)**2)
                
                if confidence >= self.discovery_threshold:
                    discovery = Discovery(
                        hypothesis=hypothesis,
                        evidence=[{"data_analysis": metrics}],
                        confidence=confidence,
                        metrics=metrics,
                        timestamp=self._get_timestamp()
                    )
                    discoveries.append(discovery)
                    self.discoveries.append(discovery)
                    logger.info(f"Discovery made with confidence {confidence:.3f}")
        
        logger.info(f"Discovery process complete: {len(discoveries)} new discoveries")
        return discoveries
    
    def get_best_discoveries(self, top_k: int = 5) -> List[Discovery]:
        """Get top discoveries by confidence"""
        sorted_discoveries = sorted(self.discoveries, key=lambda x: x.confidence, reverse=True)
        return sorted_discoveries[:top_k]
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def summary(self) -> Dict[str, Any]:
        """Get discovery engine summary statistics"""
        return {
            "total_discoveries": len(self.discoveries),
            "hypotheses_tested": self.hypotheses_tested,
            "avg_confidence": np.mean([d.confidence for d in self.discoveries]) if self.discoveries else 0.0,
            "discovery_threshold": self.discovery_threshold
        }