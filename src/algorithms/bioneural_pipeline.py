"""
Bioneural Olfactory Processing Pipeline
Complete end-to-end pipeline integrating all novel components
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass, asdict
import time

from ..models.bioneural_fusion import BioneuralOlfactoryFusion, FusionResult
from ..models.olfactory_encoder import OlfactorySignalEncoder, EncodingResult
from ..models.neural_fusion import NeuralFusionLayer, FusionOutput
from ..utils.validation import ValidationMixin
from ..utils.error_handling import robust_execution, safe_array_operation
from ..utils.performance import PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline processing result"""
    raw_signal: np.ndarray
    encoding_result: EncodingResult
    bioneural_result: FusionResult
    neural_fusion_result: FusionOutput
    final_representation: np.ndarray
    processing_time: float
    quality_metrics: Dict[str, float]
    performance_profile: Dict[str, Any]


@dataclass
class PipelineConfig:
    """Configuration for bioneural processing pipeline"""
    # Encoder configuration
    encoder_input_dim: int = 128
    encoder_output_dim: int = 256
    encoder_scales: int = 4
    encoder_spectral_bands: int = 16
    
    # Bioneural fusion configuration
    num_receptors: int = 50
    fusion_layers: int = 3
    adaptation_strength: float = 0.1
    
    # Neural fusion configuration
    fusion_dim: int = 128
    num_attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Pipeline configuration
    enable_adaptation: bool = True
    enable_profiling: bool = True
    quality_threshold: float = 0.7


class BioneuralOlfactoryPipeline(ValidationMixin):
    """
    Complete Bioneural Olfactory Processing Pipeline
    
    Integrates all novel components into a unified processing system:
    1. Olfactory signal encoding with multi-scale features
    2. Bioneural fusion with receptor ensemble modeling
    3. Multi-modal neural fusion with attention mechanisms
    4. Comprehensive quality assessment and profiling
    
    Research Contributions:
    - Novel biomimetic olfactory processing architecture
    - Multi-scale signal decomposition and fusion
    - Adaptive receptor modeling with learning
    - Attention-based cross-modal integration
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize bioneural olfactory processing pipeline
        
        Args:
            config: Pipeline configuration parameters
        """
        self.config = config or PipelineConfig()
        
        # Initialize pipeline components
        self.encoder = OlfactorySignalEncoder(
            input_dim=self.config.encoder_input_dim,
            encoding_dim=self.config.encoder_output_dim,
            num_scales=self.config.encoder_scales,
            spectral_bands=self.config.encoder_spectral_bands
        )
        
        self.bioneural_fusion = BioneuralOlfactoryFusion(
            num_receptors=self.config.num_receptors,
            signal_dim=self.config.encoder_input_dim,
            fusion_layers=self.config.fusion_layers,
            adaptation_strength=self.config.adaptation_strength
        )
        
        # Define input dimensions for neural fusion
        fusion_input_dims = {
            'encoded': self.config.encoder_output_dim,
            'bioneural': 64,  # From bioneural fusion output
            'receptors': self.config.num_receptors
        }
        
        self.neural_fusion = NeuralFusionLayer(
            input_dims=fusion_input_dims,
            fusion_dim=self.config.fusion_dim,
            num_heads=self.config.num_attention_heads,
            dropout_rate=self.config.dropout_rate
        )
        
        # Performance profiler
        self.profiler = PerformanceProfiler() if self.config.enable_profiling else None
        
        # Pipeline statistics
        self.pipeline_stats = {
            "signals_processed": 0,
            "avg_processing_time": 0.0,
            "avg_quality_score": 0.0,
            "successful_adaptations": 0,
            "quality_failures": 0
        }
        
        logger.info("BioneuralOlfactoryPipeline initialized with advanced fusion architecture")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    @safe_array_operation
    def process(self, raw_signal: np.ndarray, 
                signal_metadata: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Process raw olfactory signal through complete pipeline
        
        Args:
            raw_signal: Raw chemical signal input
            signal_metadata: Optional metadata about the signal
            
        Returns:
            PipelineResult with complete processing information
        """
        start_time = time.time()
        
        # Validate input
        if raw_signal.size == 0:
            raise ValueError("Cannot process empty signal")
        
        # Start performance profiling
        if self.profiler:
            self.profiler.start_timing("total_pipeline")
        
        try:
            # Stage 1: Signal Encoding
            if self.profiler:
                self.profiler.start_timing("encoding")
            
            encoding_result = self.encoder.encode(
                raw_signal, 
                include_quality_check=True
            )
            
            if self.profiler:
                self.profiler.end_timing("encoding")
            
            # Stage 2: Bioneural Fusion
            if self.profiler:
                self.profiler.start_timing("bioneural_fusion")
            
            bioneural_result = self.bioneural_fusion.process_signal(
                raw_signal,
                adapt=self.config.enable_adaptation
            )
            
            if self.profiler:
                self.profiler.end_timing("bioneural_fusion")
            
            # Stage 3: Prepare Multi-Modal Input
            fusion_inputs = {
                'encoded': encoding_result.encoded_signal,
                'bioneural': bioneural_result.fused_representation,
                'receptors': np.array(list(bioneural_result.receptor_activations.values()))
            }
            
            # Stage 4: Neural Fusion
            if self.profiler:
                self.profiler.start_timing("neural_fusion")
            
            neural_fusion_result = self.neural_fusion.fuse(fusion_inputs)
            
            if self.profiler:
                self.profiler.end_timing("neural_fusion")
            
            # Stage 5: Final Representation Integration
            if self.profiler:
                self.profiler.start_timing("final_integration")
            
            final_representation = self._integrate_final_representation(
                encoding_result, bioneural_result, neural_fusion_result
            )
            
            if self.profiler:
                self.profiler.end_timing("final_integration")
            
            # Stage 6: Quality Assessment
            if self.profiler:
                self.profiler.start_timing("quality_assessment")
            
            quality_metrics = self._compute_pipeline_quality(
                raw_signal, encoding_result, bioneural_result, 
                neural_fusion_result, final_representation
            )
            
            if self.profiler:
                self.profiler.end_timing("quality_assessment")
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise
        finally:
            if self.profiler:
                self.profiler.end_timing("total_pipeline")
        
        # Finalize processing
        processing_time = time.time() - start_time
        
        performance_profile = {}
        if self.profiler:
            performance_profile = self.profiler.get_timing_summary()
        
        # Update pipeline statistics
        self._update_pipeline_stats(processing_time, quality_metrics)
        
        # Check quality threshold
        overall_quality = quality_metrics.get('overall_quality', 0.0)
        if overall_quality < self.config.quality_threshold:
            self.pipeline_stats["quality_failures"] += 1
            logger.warning(f"Quality below threshold: {overall_quality:.3f} < {self.config.quality_threshold}")
        
        result = PipelineResult(
            raw_signal=raw_signal,
            encoding_result=encoding_result,
            bioneural_result=bioneural_result,
            neural_fusion_result=neural_fusion_result,
            final_representation=final_representation,
            processing_time=processing_time,
            quality_metrics=quality_metrics,
            performance_profile=performance_profile
        )
        
        logger.info(f"Pipeline processing complete: {processing_time:.4f}s, quality={overall_quality:.3f}")
        return result
    
    def _integrate_final_representation(self, encoding_result: EncodingResult,
                                      bioneural_result: FusionResult,
                                      neural_fusion_result: FusionOutput) -> np.ndarray:
        """Integrate all processing stages into final representation"""
        
        # Weighted combination of different processing levels
        weights = {
            'encoded': 0.3,      # Raw feature encoding
            'bioneural': 0.4,    # Biological modeling
            'neural_fusion': 0.3  # Attention-based fusion
        }
        
        # Normalize all components to same dimension
        target_dim = self.config.fusion_dim
        
        # Encode features
        encoded_features = encoding_result.encoded_signal
        if len(encoded_features) != target_dim:
            if len(encoded_features) < target_dim:
                encoded_features = np.pad(encoded_features, (0, target_dim - len(encoded_features)))
            else:
                encoded_features = encoded_features[:target_dim]
        
        # Bioneural features
        bioneural_features = bioneural_result.fused_representation
        if len(bioneural_features) != target_dim:
            if len(bioneural_features) < target_dim:
                bioneural_features = np.pad(bioneural_features, (0, target_dim - len(bioneural_features)))
            else:
                bioneural_features = bioneural_features[:target_dim]
        
        # Neural fusion features
        fusion_features = neural_fusion_result.fused_features
        if len(fusion_features) != target_dim:
            if len(fusion_features) < target_dim:
                fusion_features = np.pad(fusion_features, (0, target_dim - len(fusion_features)))
            else:
                fusion_features = fusion_features[:target_dim]
        
        # Weighted combination with confidence-based adjustment
        confidence_weights = np.array([
            encoding_result.quality_metrics.get('overall_quality', 0.5),
            bioneural_result.confidence_score,
            neural_fusion_result.fusion_confidence
        ])
        
        # Normalize confidence weights
        confidence_weights = confidence_weights / (np.sum(confidence_weights) + 1e-10)
        
        # Adaptive weighted combination
        final_weights = np.array([weights['encoded'], weights['bioneural'], weights['neural_fusion']])
        final_weights = 0.7 * final_weights + 0.3 * confidence_weights
        
        final_representation = (
            final_weights[0] * encoded_features +
            final_weights[1] * bioneural_features +
            final_weights[2] * fusion_features
        )
        
        return final_representation
    
    def _compute_pipeline_quality(self, raw_signal: np.ndarray,
                                encoding_result: EncodingResult,
                                bioneural_result: FusionResult,
                                neural_fusion_result: FusionOutput,
                                final_representation: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive pipeline quality metrics"""
        
        quality_metrics = {}
        
        # Individual component qualities
        quality_metrics['encoding_quality'] = encoding_result.quality_metrics.get('overall_quality', 0.0)
        quality_metrics['bioneural_confidence'] = bioneural_result.confidence_score
        quality_metrics['fusion_confidence'] = neural_fusion_result.fusion_confidence
        
        # Signal preservation metrics
        if len(raw_signal) == len(final_representation):
            preservation_corr = np.corrcoef(raw_signal, final_representation)[0, 1]
            quality_metrics['signal_preservation'] = float(preservation_corr) if not np.isnan(preservation_corr) else 0.0
        else:
            # Use partial correlation
            min_len = min(len(raw_signal), len(final_representation))
            preservation_corr = np.corrcoef(raw_signal[:min_len], final_representation[:min_len])[0, 1]
            quality_metrics['signal_preservation'] = float(preservation_corr) if not np.isnan(preservation_corr) else 0.0
        
        # Feature richness (entropy-based)
        final_abs = np.abs(final_representation)
        if np.sum(final_abs) > 0:
            final_prob = final_abs / np.sum(final_abs)
            feature_entropy = -np.sum(final_prob * np.log(final_prob + 1e-10))
            max_entropy = np.log(len(final_representation))
            quality_metrics['feature_richness'] = float(feature_entropy / max_entropy)
        else:
            quality_metrics['feature_richness'] = 0.0
        
        # Processing consistency (how well different components agree)
        component_outputs = [
            encoding_result.encoded_signal[:min(32, len(encoding_result.encoded_signal))],
            bioneural_result.fused_representation[:min(32, len(bioneural_result.fused_representation))],
            neural_fusion_result.fused_features[:min(32, len(neural_fusion_result.fused_features))]
        ]
        
        # Pad to same length for comparison
        max_comp_len = max(len(comp) for comp in component_outputs)
        padded_components = []
        for comp in component_outputs:
            if len(comp) < max_comp_len:
                comp = np.pad(comp, (0, max_comp_len - len(comp)))
            padded_components.append(comp)
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(padded_components)):
            for j in range(i+1, len(padded_components)):
                corr = np.corrcoef(padded_components[i], padded_components[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        quality_metrics['processing_consistency'] = float(np.mean(correlations)) if correlations else 0.0
        
        # Receptor diversity (from bioneural processing)
        activated_receptors = sum(1 for v in bioneural_result.receptor_activations.values() if v > 0.1)
        quality_metrics['receptor_diversity'] = float(activated_receptors / len(bioneural_result.receptor_activations))
        
        # Pattern complexity
        quality_metrics['pattern_complexity'] = bioneural_result.pattern_complexity
        
        # Modality fusion effectiveness
        modality_contributions = neural_fusion_result.modality_contributions
        contribution_balance = 1.0 - np.var(list(modality_contributions.values()))  # Higher balance = better
        quality_metrics['fusion_effectiveness'] = float(np.clip(contribution_balance, 0.0, 1.0))
        
        # Overall quality score (weighted combination)
        quality_components = [
            quality_metrics['encoding_quality'] * 0.2,
            quality_metrics['bioneural_confidence'] * 0.2,
            quality_metrics['fusion_confidence'] * 0.15,
            quality_metrics['signal_preservation'] * 0.15,
            quality_metrics['feature_richness'] * 0.1,
            quality_metrics['processing_consistency'] * 0.1,
            quality_metrics['receptor_diversity'] * 0.05,
            quality_metrics['fusion_effectiveness'] * 0.05
        ]
        
        quality_metrics['overall_quality'] = float(np.sum(quality_components))
        
        return quality_metrics
    
    def _update_pipeline_stats(self, processing_time: float, quality_metrics: Dict[str, float]):
        """Update pipeline performance statistics"""
        self.pipeline_stats["signals_processed"] += 1
        
        # Update average processing time
        total_processed = self.pipeline_stats["signals_processed"]
        self.pipeline_stats["avg_processing_time"] = (
            (self.pipeline_stats["avg_processing_time"] * (total_processed - 1) + 
             processing_time) / total_processed
        )
        
        # Update average quality score
        overall_quality = quality_metrics.get('overall_quality', 0.0)
        self.pipeline_stats["avg_quality_score"] = (
            (self.pipeline_stats["avg_quality_score"] * (total_processed - 1) + 
             overall_quality) / total_processed
        )
        
        # Count successful adaptations (if bioneural component adapted)
        if self.config.enable_adaptation and overall_quality > self.config.quality_threshold:
            self.pipeline_stats["successful_adaptations"] += 1
    
    def batch_process(self, signals: np.ndarray, 
                     batch_metadata: Optional[List[Dict[str, Any]]] = None) -> List[PipelineResult]:
        """Process multiple signals efficiently"""
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        
        results = []
        metadata_list = batch_metadata or [None] * signals.shape[0]
        
        for i in range(signals.shape[0]):
            metadata = metadata_list[i] if i < len(metadata_list) else None
            result = self.process(signals[i], metadata)
            results.append(result)
        
        return results
    
    def benchmark_performance(self, test_signals: np.ndarray, 
                            num_iterations: int = 10) -> Dict[str, Any]:
        """Comprehensive performance benchmarking"""
        logger.info(f"Starting performance benchmark with {num_iterations} iterations")
        
        benchmark_results = {
            "num_iterations": num_iterations,
            "num_test_signals": len(test_signals),
            "processing_times": [],
            "quality_scores": [],
            "component_timings": {},
            "memory_usage": []
        }
        
        for iteration in range(num_iterations):
            iteration_times = []
            iteration_qualities = []
            
            for signal in test_signals:
                result = self.process(signal)
                iteration_times.append(result.processing_time)
                iteration_qualities.append(result.quality_metrics['overall_quality'])
                
                # Collect component timings
                if result.performance_profile:
                    for component, timing in result.performance_profile.items():
                        if component not in benchmark_results["component_timings"]:
                            benchmark_results["component_timings"][component] = []
                        benchmark_results["component_timings"][component].append(timing)
            
            benchmark_results["processing_times"].extend(iteration_times)
            benchmark_results["quality_scores"].extend(iteration_qualities)
        
        # Compute statistics
        times = benchmark_results["processing_times"]
        qualities = benchmark_results["quality_scores"]
        
        benchmark_results["performance_stats"] = {
            "avg_processing_time": np.mean(times),
            "std_processing_time": np.std(times),
            "min_processing_time": np.min(times),
            "max_processing_time": np.max(times),
            "avg_quality_score": np.mean(qualities),
            "std_quality_score": np.std(qualities),
            "min_quality_score": np.min(qualities),
            "max_quality_score": np.max(qualities)
        }
        
        # Component timing statistics
        for component, timings in benchmark_results["component_timings"].items():
            benchmark_results["component_timings"][component] = {
                "avg": np.mean(timings),
                "std": np.std(timings),
                "min": np.min(timings),
                "max": np.max(timings)
            }
        
        logger.info(f"Benchmark complete: avg_time={benchmark_results['performance_stats']['avg_processing_time']:.4f}s, avg_quality={benchmark_results['performance_stats']['avg_quality_score']:.3f}")
        return benchmark_results
    
    def get_component_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries from all pipeline components"""
        return {
            "encoder": self.encoder.summary(),
            "bioneural_fusion": self.bioneural_fusion.summary(),
            "neural_fusion": self.neural_fusion.summary(),
            "pipeline": {
                "config": asdict(self.config),
                "stats": self.pipeline_stats.copy()
            }
        }
    
    def export_processing_result(self, result: PipelineResult) -> Dict[str, Any]:
        """Export processing result in serializable format"""
        return {
            "raw_signal_shape": result.raw_signal.shape,
            "raw_signal_stats": {
                "mean": float(np.mean(result.raw_signal)),
                "std": float(np.std(result.raw_signal)),
                "min": float(np.min(result.raw_signal)),
                "max": float(np.max(result.raw_signal))
            },
            "encoding_result": {
                "encoded_shape": result.encoding_result.encoded_signal.shape,
                "molecular_descriptors": result.encoding_result.molecular_descriptors,
                "quality_metrics": result.encoding_result.quality_metrics
            },
            "bioneural_result": {
                "fused_shape": result.bioneural_result.fused_representation.shape,
                "confidence_score": result.bioneural_result.confidence_score,
                "pattern_complexity": result.bioneural_result.pattern_complexity,
                "num_active_receptors": len([v for v in result.bioneural_result.receptor_activations.values() if v > 0.1])
            },
            "neural_fusion_result": {
                "fused_shape": result.neural_fusion_result.fused_features.shape,
                "fusion_confidence": result.neural_fusion_result.fusion_confidence,
                "modality_contributions": result.neural_fusion_result.modality_contributions
            },
            "final_representation_shape": result.final_representation.shape,
            "processing_time": result.processing_time,
            "quality_metrics": result.quality_metrics,
            "performance_profile": result.performance_profile
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        return {
            "pipeline_type": "BioneuralOlfactoryPipeline",
            "configuration": asdict(self.config),
            "component_summaries": self.get_component_summaries(),
            "pipeline_statistics": self.pipeline_stats.copy(),
            "research_contributions": [
                "Novel biomimetic olfactory processing architecture",
                "Multi-scale signal decomposition and fusion",
                "Adaptive receptor modeling with learning",
                "Attention-based cross-modal integration",
                "Comprehensive quality assessment framework"
            ]
        }