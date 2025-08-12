"""
Baseline Model Suite for Comparative Analysis
Standard approaches for olfactory signal processing comparison
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from ..utils.validation import ValidationMixin
from ..utils.error_handling import robust_execution, safe_array_operation

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from baseline model processing"""
    processed_signal: np.ndarray
    processing_time: float
    model_name: str
    performance_metrics: Dict[str, float]


class BaselineModel(ABC, ValidationMixin):
    """Abstract base class for baseline models"""
    
    @abstractmethod
    def process(self, signal: np.ndarray) -> BaselineResult:
        """Process input signal and return result"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get model name"""
        pass


class PCABaseline(BaselineModel):
    """Principal Component Analysis baseline"""
    
    def __init__(self, n_components: int = 64, whiten: bool = True):
        self.n_components = self.validate_positive_int(n_components, "n_components")
        self.whiten = whiten
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.fitted = False
        
    def fit(self, signals: np.ndarray):
        """Fit PCA to training signals"""
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        
        # Center the data
        self.mean_ = np.mean(signals, axis=0)
        centered_signals = signals - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_signals.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        if self.whiten:
            self.components_ = self.components_ / np.sqrt(self.explained_variance_[:, np.newaxis])
        
        self.fitted = True
        logger.info(f"PCA fitted with {self.n_components} components, explained variance ratio: {np.sum(self.explained_variance_) / np.sum(eigenvalues):.3f}")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    @safe_array_operation
    def process(self, signal: np.ndarray) -> BaselineResult:
        """Process signal using PCA transformation"""
        start_time = time.time()
        
        if not self.fitted:
            # Auto-fit on single signal (not ideal but for comparison)
            self.fit(signal.reshape(1, -1))
        
        # Center the signal
        centered_signal = signal - self.mean_
        
        # Transform
        transformed = np.dot(centered_signal, self.components_.T)
        
        processing_time = time.time() - start_time
        
        # Performance metrics
        reconstruction = np.dot(transformed, self.components_) + self.mean_
        reconstruction_error = np.mean((signal - reconstruction) ** 2)
        
        metrics = {
            "reconstruction_error": float(reconstruction_error),
            "variance_explained": float(np.sum(self.explained_variance_) / np.sum(self.explained_variance_)) if self.explained_variance_ is not None else 0.0,
            "compression_ratio": float(len(signal) / len(transformed)),
            "signal_to_noise_ratio": float(np.var(signal) / (reconstruction_error + 1e-10))
        }
        
        return BaselineResult(
            processed_signal=transformed,
            processing_time=processing_time,
            model_name=self.get_name(),
            performance_metrics=metrics
        )
    
    def get_name(self) -> str:
        return f"PCA_{self.n_components}{'_whitened' if self.whiten else ''}"


class ICABaseline(BaselineModel):
    """Independent Component Analysis baseline"""
    
    def __init__(self, n_components: int = 64, max_iter: int = 200, tol: float = 1e-4):
        self.n_components = self.validate_positive_int(n_components, "n_components")
        self.max_iter = max_iter
        self.tol = tol
        self.components_ = None
        self.mean_ = None
        self.whitening_ = None
        self.fitted = False
    
    def _g(self, x):
        """Non-linearity for FastICA"""
        return np.tanh(x)
    
    def _g_deriv(self, x):
        """Derivative of non-linearity"""
        return 1 - np.tanh(x) ** 2
    
    def fit(self, signals: np.ndarray):
        """Fit ICA to training signals"""
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        
        # Center the data
        self.mean_ = np.mean(signals, axis=0)
        centered_signals = signals - self.mean_
        
        # Whiten the data using PCA
        cov_matrix = np.cov(centered_signals.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Whitening matrix
        d = np.sqrt(eigenvalues[:self.n_components])
        self.whitening_ = (eigenvectors[:, :self.n_components] / d).T
        
        # Whiten the data
        X_white = np.dot(self.whitening_, centered_signals.T).T
        
        # FastICA algorithm
        W = np.random.randn(self.n_components, self.n_components)
        W = np.linalg.qr(W)[0]  # Orthogonalize
        
        for iteration in range(self.max_iter):
            W_old = W.copy()
            
            # Update each component
            for i in range(self.n_components):
                w = W[i, :]
                
                # Update rule
                g_wx = self._g(np.dot(X_white, w))
                g_deriv_wx = self._g_deriv(np.dot(X_white, w))
                
                w_new = np.mean(X_white.T * g_wx, axis=1) - np.mean(g_deriv_wx) * w
                
                # Orthogonalize
                for j in range(i):
                    w_new -= np.dot(w_new, W[j, :]) * W[j, :]
                
                # Normalize
                w_new = w_new / (np.linalg.norm(w_new) + 1e-10)
                W[i, :] = w_new
            
            # Check convergence
            if np.max(np.abs(np.abs(np.diag(np.dot(W, W_old.T))) - 1)) < self.tol:
                break
        
        self.components_ = np.dot(W, self.whitening_)
        self.fitted = True
        logger.info(f"ICA fitted with {self.n_components} components after {iteration+1} iterations")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    @safe_array_operation
    def process(self, signal: np.ndarray) -> BaselineResult:
        """Process signal using ICA transformation"""
        start_time = time.time()
        
        if not self.fitted:
            # Auto-fit on single signal
            self.fit(signal.reshape(1, -1))
        
        # Center and transform
        centered_signal = signal - self.mean_
        transformed = np.dot(centered_signal, self.components_.T)
        
        processing_time = time.time() - start_time
        
        # Performance metrics
        reconstruction = np.dot(transformed, self.components_) + self.mean_
        reconstruction_error = np.mean((signal - reconstruction) ** 2)
        
        # Independence measure (simplified)
        independence_score = float(np.mean(np.abs(np.corrcoef(transformed.reshape(1, -1) if transformed.ndim == 1 else transformed))))
        
        metrics = {
            "reconstruction_error": float(reconstruction_error),
            "independence_score": independence_score,
            "compression_ratio": float(len(signal) / len(transformed)),
            "signal_to_noise_ratio": float(np.var(signal) / (reconstruction_error + 1e-10))
        }
        
        return BaselineResult(
            processed_signal=transformed,
            processing_time=processing_time,
            model_name=self.get_name(),
            performance_metrics=metrics
        )
    
    def get_name(self) -> str:
        return f"ICA_{self.n_components}"


class FFTBaseline(BaselineModel):
    """Fast Fourier Transform baseline"""
    
    def __init__(self, keep_ratio: float = 0.5, use_magnitude_only: bool = False):
        self.keep_ratio = self.validate_probability(keep_ratio, "keep_ratio")
        self.use_magnitude_only = use_magnitude_only
    
    @robust_execution(recovery_strategy='graceful_degradation')
    @safe_array_operation
    def process(self, signal: np.ndarray) -> BaselineResult:
        """Process signal using FFT transformation"""
        start_time = time.time()
        
        # Compute FFT
        fft_signal = np.fft.fft(signal)
        
        # Keep only a fraction of coefficients (low frequencies)
        keep_coeffs = int(len(fft_signal) * self.keep_ratio)
        
        if self.use_magnitude_only:
            # Use only magnitude spectrum
            processed_signal = np.abs(fft_signal[:keep_coeffs])
        else:
            # Keep both real and imaginary parts
            kept_fft = np.zeros_like(fft_signal)
            kept_fft[:keep_coeffs] = fft_signal[:keep_coeffs]
            kept_fft[-keep_coeffs:] = fft_signal[-keep_coeffs:]  # Maintain symmetry
            processed_signal = np.concatenate([np.real(kept_fft[:keep_coeffs]), 
                                             np.imag(kept_fft[:keep_coeffs])])
        
        processing_time = time.time() - start_time
        
        # Performance metrics
        if not self.use_magnitude_only:
            # Reconstruction from kept coefficients
            kept_fft = np.zeros(len(signal), dtype=complex)
            mid = len(processed_signal) // 2
            kept_fft[:keep_coeffs] = processed_signal[:mid] + 1j * processed_signal[mid:]
            kept_fft[-keep_coeffs:] = np.conj(kept_fft[keep_coeffs-1:0:-1])
            reconstruction = np.real(np.fft.ifft(kept_fft))
            reconstruction_error = np.mean((signal - reconstruction) ** 2)
        else:
            reconstruction_error = float('inf')  # Cannot reconstruct from magnitude only
        
        # Frequency domain metrics
        spectral_centroid = np.sum(np.arange(keep_coeffs) * np.abs(fft_signal[:keep_coeffs])) / (np.sum(np.abs(fft_signal[:keep_coeffs])) + 1e-10)
        spectral_bandwidth = np.sqrt(np.sum(((np.arange(keep_coeffs) - spectral_centroid) ** 2) * np.abs(fft_signal[:keep_coeffs])) / (np.sum(np.abs(fft_signal[:keep_coeffs])) + 1e-10))
        
        metrics = {
            "reconstruction_error": float(reconstruction_error),
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "compression_ratio": float(len(signal) / len(processed_signal)),
            "frequency_resolution": float(keep_coeffs / len(signal))
        }
        
        return BaselineResult(
            processed_signal=processed_signal,
            processing_time=processing_time,
            model_name=self.get_name(),
            performance_metrics=metrics
        )
    
    def get_name(self) -> str:
        return f"FFT_{self.keep_ratio}{'_mag' if self.use_magnitude_only else '_complex'}"


class WaveletBaseline(BaselineModel):
    """Discrete Wavelet Transform baseline"""
    
    def __init__(self, wavelet_type: str = 'db4', levels: int = 4, keep_ratio: float = 0.5):
        self.wavelet_type = wavelet_type
        self.levels = self.validate_positive_int(levels, "levels")
        self.keep_ratio = self.validate_probability(keep_ratio, "keep_ratio")
    
    def _dwt_1d(self, signal: np.ndarray) -> List[np.ndarray]:
        """Simple 1D Discrete Wavelet Transform implementation"""
        # Simplified Haar wavelet for demonstration
        # In practice, would use more sophisticated wavelets
        
        coefficients = []
        current_signal = signal.copy()
        
        for level in range(self.levels):
            if len(current_signal) < 2:
                break
            
            # Ensure even length
            if len(current_signal) % 2 == 1:
                current_signal = np.append(current_signal, current_signal[-1])
            
            # Haar wavelet decomposition
            approx = (current_signal[::2] + current_signal[1::2]) / np.sqrt(2)
            detail = (current_signal[::2] - current_signal[1::2]) / np.sqrt(2)
            
            coefficients.append(detail)
            current_signal = approx
        
        coefficients.append(current_signal)  # Final approximation
        return coefficients[::-1]  # Reverse order (approximation first)
    
    @robust_execution(recovery_strategy='graceful_degradation')
    @safe_array_operation
    def process(self, signal: np.ndarray) -> BaselineResult:
        """Process signal using wavelet transformation"""
        start_time = time.time()
        
        # Compute wavelet coefficients
        coefficients = self._dwt_1d(signal)
        
        # Flatten all coefficients
        all_coeffs = np.concatenate(coefficients)
        
        # Keep only top coefficients by magnitude
        keep_count = int(len(all_coeffs) * self.keep_ratio)
        indices = np.argsort(np.abs(all_coeffs))[::-1][:keep_count]
        
        # Create sparse representation
        processed_signal = np.zeros_like(all_coeffs)
        processed_signal[indices] = all_coeffs[indices]
        
        processing_time = time.time() - start_time
        
        # Performance metrics
        sparsity = float(np.sum(processed_signal != 0) / len(processed_signal))
        energy_preserved = float(np.sum(processed_signal ** 2) / (np.sum(all_coeffs ** 2) + 1e-10))
        
        # Compute wavelet entropy
        coeffs_abs = np.abs(all_coeffs)
        coeffs_prob = coeffs_abs / (np.sum(coeffs_abs) + 1e-10)
        wavelet_entropy = -np.sum(coeffs_prob * np.log(coeffs_prob + 1e-10))
        
        metrics = {
            "sparsity": sparsity,
            "energy_preserved": energy_preserved,
            "wavelet_entropy": float(wavelet_entropy),
            "compression_ratio": float(len(signal) / keep_count),
            "detail_coeffs_ratio": float(sum(len(coefficients[i]) for i in range(1, len(coefficients))) / len(all_coeffs))
        }
        
        return BaselineResult(
            processed_signal=processed_signal[:keep_count],  # Return only kept coefficients
            processing_time=processing_time,
            model_name=self.get_name(),
            performance_metrics=metrics
        )
    
    def get_name(self) -> str:
        return f"Wavelet_{self.wavelet_type}_{self.levels}L_{self.keep_ratio}"


class RandomProjectionBaseline(BaselineModel):
    """Random projection baseline (Johnson-Lindenstrauss)"""
    
    def __init__(self, target_dim: int = 64, distribution: str = 'gaussian'):
        self.target_dim = self.validate_positive_int(target_dim, "target_dim")
        self.distribution = distribution
        self.projection_matrix = None
    
    def _initialize_projection(self, input_dim: int):
        """Initialize random projection matrix"""
        if self.distribution == 'gaussian':
            self.projection_matrix = np.random.randn(input_dim, self.target_dim) / np.sqrt(self.target_dim)
        elif self.distribution == 'sparse':
            # Sparse random projection (Achlioptas)
            s = np.sqrt(input_dim)
            prob_nonzero = 1 / s
            self.projection_matrix = np.random.choice(
                [-np.sqrt(s), 0, np.sqrt(s)], 
                size=(input_dim, self.target_dim),
                p=[prob_nonzero/2, 1-prob_nonzero, prob_nonzero/2]
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
    
    @robust_execution(recovery_strategy='graceful_degradation')
    @safe_array_operation
    def process(self, signal: np.ndarray) -> BaselineResult:
        """Process signal using random projection"""
        start_time = time.time()
        
        if self.projection_matrix is None or self.projection_matrix.shape[0] != len(signal):
            self._initialize_projection(len(signal))
        
        # Project signal
        processed_signal = np.dot(signal, self.projection_matrix)
        
        processing_time = time.time() - start_time
        
        # Performance metrics
        # Estimate distance preservation (cannot compute exactly without pairs)
        original_norm = np.linalg.norm(signal)
        projected_norm = np.linalg.norm(processed_signal)
        norm_preservation = float(projected_norm / (original_norm + 1e-10))
        
        # Random projection properties
        jl_bound = np.sqrt(8 * np.log(2) / self.target_dim)  # Johnson-Lindenstrauss bound
        compression_ratio = float(len(signal) / self.target_dim)
        
        metrics = {
            "norm_preservation": norm_preservation,
            "jl_bound": float(jl_bound),
            "compression_ratio": compression_ratio,
            "projection_density": float(np.sum(self.projection_matrix != 0) / self.projection_matrix.size),
            "target_dimension": float(self.target_dim)
        }
        
        return BaselineResult(
            processed_signal=processed_signal,
            processing_time=processing_time,
            model_name=self.get_name(),
            performance_metrics=metrics
        )
    
    def get_name(self) -> str:
        return f"RandomProj_{self.target_dim}_{self.distribution}"


class BaselineModelSuite(ValidationMixin):
    """Suite of baseline models for comprehensive comparison"""
    
    def __init__(self, signal_dim: int = 128):
        self.signal_dim = self.validate_positive_int(signal_dim, "signal_dim")
        self.models = self._initialize_baseline_models()
        self.fitted_models = set()
        
    def _initialize_baseline_models(self) -> Dict[str, BaselineModel]:
        """Initialize all baseline models"""
        target_dim = min(64, self.signal_dim // 2)
        
        models = {
            'pca_standard': PCABaseline(n_components=target_dim, whiten=False),
            'pca_whitened': PCABaseline(n_components=target_dim, whiten=True),
            'ica_standard': ICABaseline(n_components=target_dim),
            'fft_complex': FFTBaseline(keep_ratio=0.5, use_magnitude_only=False),
            'fft_magnitude': FFTBaseline(keep_ratio=0.5, use_magnitude_only=True),
            'wavelet_haar': WaveletBaseline(wavelet_type='haar', levels=4, keep_ratio=0.5),
            'random_proj_gaussian': RandomProjectionBaseline(target_dim=target_dim, distribution='gaussian'),
            'random_proj_sparse': RandomProjectionBaseline(target_dim=target_dim, distribution='sparse')
        }
        
        return models
    
    def fit_models(self, training_signals: np.ndarray):
        """Fit baseline models that require training"""
        if training_signals.ndim == 1:
            training_signals = training_signals.reshape(1, -1)
        
        fittable_models = ['pca_standard', 'pca_whitened', 'ica_standard']
        
        for model_name in fittable_models:
            if model_name in self.models:
                logger.info(f"Fitting {model_name}")
                try:
                    self.models[model_name].fit(training_signals)
                    self.fitted_models.add(model_name)
                except Exception as e:
                    logger.error(f"Failed to fit {model_name}: {e}")
    
    def process_signal(self, signal: np.ndarray) -> Dict[str, BaselineResult]:
        """Process signal with all baseline models"""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                result = model.process(signal)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to process with {model_name}: {e}")
                # Create dummy result for failed models
                results[model_name] = BaselineResult(
                    processed_signal=np.zeros(1),
                    processing_time=float('inf'),
                    model_name=model_name,
                    performance_metrics={"error": float('inf')}
                )
        
        return results
    
    def benchmark_suite(self, test_signals: np.ndarray, 
                       training_signals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark of all baseline models"""
        if training_signals is not None:
            self.fit_models(training_signals)
        
        if test_signals.ndim == 1:
            test_signals = test_signals.reshape(1, -1)
        
        benchmark_results = {
            "num_test_signals": test_signals.shape[0],
            "signal_dimension": test_signals.shape[1],
            "model_results": {},
            "comparative_metrics": {}
        }
        
        all_results = []
        
        # Process each test signal
        for i, signal in enumerate(test_signals):
            signal_results = self.process_signal(signal)
            all_results.append(signal_results)
        
        # Aggregate results by model
        for model_name in self.models.keys():
            model_times = []
            model_metrics = {}
            
            for signal_results in all_results:
                if model_name in signal_results:
                    result = signal_results[model_name]
                    model_times.append(result.processing_time)
                    
                    # Aggregate metrics
                    for metric_name, metric_value in result.performance_metrics.items():
                        if metric_name not in model_metrics:
                            model_metrics[metric_name] = []
                        model_metrics[metric_name].append(metric_value)
            
            # Compute statistics
            benchmark_results["model_results"][model_name] = {
                "processing_times": {
                    "mean": float(np.mean(model_times)) if model_times else float('inf'),
                    "std": float(np.std(model_times)) if model_times else 0.0,
                    "min": float(np.min(model_times)) if model_times else float('inf'),
                    "max": float(np.max(model_times)) if model_times else float('inf')
                },
                "performance_metrics": {}
            }
            
            for metric_name, metric_values in model_metrics.items():
                if metric_values and not np.all(np.isinf(metric_values)):
                    benchmark_results["model_results"][model_name]["performance_metrics"][metric_name] = {
                        "mean": float(np.mean(metric_values)),
                        "std": float(np.std(metric_values)),
                        "min": float(np.min(metric_values)),
                        "max": float(np.max(metric_values))
                    }
        
        # Comparative analysis
        processing_times = {name: results["processing_times"]["mean"] 
                          for name, results in benchmark_results["model_results"].items()}
        
        benchmark_results["comparative_metrics"] = {
            "fastest_model": min(processing_times.keys(), key=lambda k: processing_times[k]),
            "slowest_model": max(processing_times.keys(), key=lambda k: processing_times[k]),
            "speed_range": {
                "fastest_time": min(processing_times.values()),
                "slowest_time": max(processing_times.values()),
                "speedup_factor": max(processing_times.values()) / (min(processing_times.values()) + 1e-10)
            }
        }
        
        logger.info(f"Baseline suite benchmark complete: {len(self.models)} models, {len(test_signals)} signals")
        return benchmark_results
    
    def get_model_summaries(self) -> Dict[str, str]:
        """Get summary descriptions of all baseline models"""
        return {name: model.get_name() for name, model in self.models.items()}
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive baseline suite summary"""
        return {
            "suite_type": "BaselineModelSuite",
            "signal_dimension": self.signal_dim,
            "num_models": len(self.models),
            "fitted_models": list(self.fitted_models),
            "model_descriptions": self.get_model_summaries(),
            "model_categories": {
                "dimensionality_reduction": ["pca_standard", "pca_whitened", "ica_standard", "random_proj_gaussian", "random_proj_sparse"],
                "frequency_domain": ["fft_complex", "fft_magnitude"],
                "time_frequency": ["wavelet_haar"],
                "random_projection": ["random_proj_gaussian", "random_proj_sparse"]
            }
        }