"""
GPU Acceleration Framework
CUDA/OpenCL acceleration for bioneural olfactory fusion processing
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.validation import ValidationMixin
from ..utils.error_handling import robust_execution, safe_array_operation

logger = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration"""
    device_type: str = 'cuda'  # 'cuda' or 'opencl'
    device_id: int = 0
    memory_pool: bool = True
    precision: str = 'float32'  # 'float32' or 'float64'
    batch_size: int = 32
    use_streams: bool = True


@dataclass
class GPUMemoryInfo:
    """GPU memory information"""
    total_memory: int
    free_memory: int
    used_memory: int
    memory_utilization: float


@dataclass
class GPUPerformanceMetrics:
    """GPU performance metrics"""
    kernel_execution_time: float
    memory_transfer_time: float
    total_processing_time: float
    gpu_utilization: float
    memory_bandwidth: float
    flops_per_second: float


class GPUKernel(ABC):
    """Abstract base class for GPU kernels"""
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute GPU kernel"""
        pass
    
    @abstractmethod
    def get_flops(self) -> int:
        """Get floating point operations count"""
        pass


class CUDAReceptorKernel(GPUKernel):
    """CUDA kernel for receptor ensemble processing"""
    
    def __init__(self, num_receptors: int, signal_dim: int):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for CUDA acceleration")
        
        self.num_receptors = num_receptors
        self.signal_dim = signal_dim
        
        # CUDA kernel code for receptor processing
        self.kernel_code = """
        extern "C" __global__ void process_receptors(
            const float* signal,
            const float* sensitivity_profiles,
            const float* binding_affinities,
            const float* response_thresholds,
            float* activations,
            int signal_dim,
            int num_receptors
        ) {
            int receptor_id = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (receptor_id >= num_receptors) return;
            
            // Compute binding strength (dot product)
            float binding_strength = 0.0f;
            for (int i = 0; i < signal_dim; i++) {
                binding_strength += signal[i] * sensitivity_profiles[receptor_id * signal_dim + i];
            }
            
            // Scale by binding affinity
            float scaled_binding = binding_strength * binding_affinities[receptor_id];
            
            // Apply threshold and nonlinearity
            float threshold = response_thresholds[receptor_id];
            if (scaled_binding > threshold) {
                // Sigmoid activation: 1 / (1 + exp(-10 * (x - threshold)))
                float x = 10.0f * (scaled_binding - threshold);
                activations[receptor_id] = 1.0f / (1.0f + expf(-x));
            } else {
                activations[receptor_id] = 0.0f;
            }
        }
        """
        
        # Compile kernel
        self.kernel = cp.RawKernel(self.kernel_code, 'process_receptors')
        
        # Calculate grid and block dimensions
        self.threads_per_block = 256
        self.blocks_per_grid = (num_receptors + self.threads_per_block - 1) // self.threads_per_block
    
    def execute(self, 
                signal: np.ndarray,
                sensitivity_profiles: np.ndarray,
                binding_affinities: np.ndarray,
                response_thresholds: np.ndarray) -> np.ndarray:
        """Execute receptor processing kernel"""
        
        # Transfer data to GPU
        signal_gpu = cp.asarray(signal, dtype=cp.float32)
        profiles_gpu = cp.asarray(sensitivity_profiles, dtype=cp.float32)
        affinities_gpu = cp.asarray(binding_affinities, dtype=cp.float32)
        thresholds_gpu = cp.asarray(response_thresholds, dtype=cp.float32)
        
        # Allocate output
        activations_gpu = cp.zeros(self.num_receptors, dtype=cp.float32)
        
        # Execute kernel
        self.kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (signal_gpu, profiles_gpu, affinities_gpu, thresholds_gpu, activations_gpu,
             self.signal_dim, self.num_receptors)
        )
        
        # Transfer result back to CPU
        return cp.asnumpy(activations_gpu)
    
    def get_flops(self) -> int:
        """Calculate floating point operations"""
        # Dot product: signal_dim multiplications + signal_dim-1 additions per receptor
        # Plus sigmoid computation per receptor
        return self.num_receptors * (2 * self.signal_dim + 10)  # Approximate


class CUDAFusionKernel(GPUKernel):
    """CUDA kernel for neural fusion processing"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for CUDA acceleration")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        # CUDA kernel for multi-head attention
        self.attention_kernel_code = """
        extern "C" __global__ void multi_head_attention(
            const float* input,
            const float* query_weights,
            const float* key_weights, 
            const float* value_weights,
            float* output,
            int batch_size,
            int input_dim,
            int head_dim,
            int num_heads
        ) {
            int head_id = blockIdx.x;
            int batch_id = blockIdx.y;
            int dim_id = threadIdx.x;
            
            if (head_id >= num_heads || batch_id >= batch_size || dim_id >= head_dim) return;
            
            // Compute attention for this head
            int head_offset = head_id * input_dim * head_dim;
            int batch_offset = batch_id * input_dim;
            
            // Query computation
            float query = 0.0f;
            for (int i = 0; i < input_dim; i++) {
                query += input[batch_offset + i] * query_weights[head_offset + i * head_dim + dim_id];
            }
            
            // Key computation (simplified self-attention)
            float key = query; // Self-attention
            
            // Value computation
            float value = 0.0f;
            for (int i = 0; i < input_dim; i++) {
                value += input[batch_offset + i] * value_weights[head_offset + i * head_dim + dim_id];
            }
            
            // Attention score (simplified)
            float attention_score = query * key / sqrtf((float)head_dim);
            float attention_weight = 1.0f / (1.0f + expf(-attention_score)); // Sigmoid
            
            // Output
            int output_offset = batch_id * num_heads * head_dim + head_id * head_dim + dim_id;
            output[output_offset] = attention_weight * value;
        }
        """
        
        self.kernel = cp.RawKernel(self.attention_kernel_code, 'multi_head_attention')
    
    def execute(self, 
                input_tensor: np.ndarray,
                query_weights: np.ndarray,
                key_weights: np.ndarray, 
                value_weights: np.ndarray) -> np.ndarray:
        """Execute fusion kernel"""
        
        batch_size = input_tensor.shape[0] if input_tensor.ndim > 1 else 1
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.reshape(1, -1)
        
        # Transfer to GPU
        input_gpu = cp.asarray(input_tensor, dtype=cp.float32)
        query_gpu = cp.asarray(query_weights, dtype=cp.float32)
        key_gpu = cp.asarray(key_weights, dtype=cp.float32)
        value_gpu = cp.asarray(value_weights, dtype=cp.float32)
        
        # Allocate output
        output_gpu = cp.zeros((batch_size, self.output_dim), dtype=cp.float32)
        
        # Execute kernel
        grid_dim = (self.num_heads, batch_size)
        block_dim = (self.head_dim,)
        
        self.kernel(
            grid_dim, block_dim,
            (input_gpu, query_gpu, key_gpu, value_gpu, output_gpu,
             batch_size, self.input_dim, self.head_dim, self.num_heads)
        )
        
        return cp.asnumpy(output_gpu)
    
    def get_flops(self) -> int:
        """Calculate floating point operations"""
        # Approximate FLOPS for multi-head attention
        return self.num_heads * self.input_dim * self.head_dim * 6


class OpenCLReceptorKernel(GPUKernel):
    """OpenCL kernel for receptor ensemble processing"""
    
    def __init__(self, num_receptors: int, signal_dim: int):
        if not OPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL not available for OpenCL acceleration")
        
        self.num_receptors = num_receptors
        self.signal_dim = signal_dim
        
        # Initialize OpenCL context
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        
        # OpenCL kernel code
        kernel_code = """
        __kernel void process_receptors(
            __global const float* signal,
            __global const float* sensitivity_profiles,
            __global const float* binding_affinities,
            __global const float* response_thresholds,
            __global float* activations,
            const int signal_dim,
            const int num_receptors
        ) {
            int receptor_id = get_global_id(0);
            
            if (receptor_id >= num_receptors) return;
            
            // Compute binding strength
            float binding_strength = 0.0f;
            for (int i = 0; i < signal_dim; i++) {
                binding_strength += signal[i] * sensitivity_profiles[receptor_id * signal_dim + i];
            }
            
            // Scale by binding affinity
            float scaled_binding = binding_strength * binding_affinities[receptor_id];
            
            // Apply threshold and sigmoid
            float threshold = response_thresholds[receptor_id];
            if (scaled_binding > threshold) {
                float x = 10.0f * (scaled_binding - threshold);
                activations[receptor_id] = 1.0f / (1.0f + exp(-x));
            } else {
                activations[receptor_id] = 0.0f;
            }
        }
        """
        
        # Build program
        self.program = cl.Program(self.ctx, kernel_code).build()
        self.kernel = self.program.process_receptors
    
    def execute(self,
                signal: np.ndarray,
                sensitivity_profiles: np.ndarray,
                binding_affinities: np.ndarray,
                response_thresholds: np.ndarray) -> np.ndarray:
        """Execute OpenCL kernel"""
        
        # Create buffers
        signal_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                              hostbuf=signal.astype(np.float32))
        profiles_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=sensitivity_profiles.astype(np.float32))
        affinities_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                 hostbuf=binding_affinities.astype(np.float32))
        thresholds_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                 hostbuf=response_thresholds.astype(np.float32))
        
        # Output buffer
        activations = np.zeros(self.num_receptors, dtype=np.float32)
        activations_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, activations.nbytes)
        
        # Execute kernel
        global_size = (self.num_receptors,)
        self.kernel(self.queue, global_size, None,
                   signal_buf, profiles_buf, affinities_buf, thresholds_buf, activations_buf,
                   np.int32(self.signal_dim), np.int32(self.num_receptors))
        
        # Read result
        cl.enqueue_copy(self.queue, activations, activations_buf)
        
        return activations
    
    def get_flops(self) -> int:
        """Calculate floating point operations"""
        return self.num_receptors * (2 * self.signal_dim + 10)


class GPUAcceleratedBioneuralFusion(ValidationMixin):
    """
    GPU-Accelerated Bioneural Olfactory Fusion
    
    Provides GPU acceleration for computationally intensive components:
    1. CUDA acceleration with CuPy for NVIDIA GPUs
    2. OpenCL acceleration for cross-platform GPU support
    3. Automatic fallback to CPU processing
    4. Memory management and optimization
    5. Performance monitoring and profiling
    """
    
    def __init__(self, 
                 num_receptors: int = 50,
                 signal_dim: int = 128,
                 fusion_dim: int = 128,
                 num_heads: int = 8,
                 gpu_config: Optional[GPUConfig] = None):
        """
        Initialize GPU-accelerated bioneural fusion
        
        Args:
            num_receptors: Number of olfactory receptors
            signal_dim: Input signal dimension
            fusion_dim: Fusion layer dimension
            num_heads: Number of attention heads
            gpu_config: GPU configuration
        """
        self.num_receptors = self.validate_positive_int(num_receptors, "num_receptors")
        self.signal_dim = self.validate_positive_int(signal_dim, "signal_dim")
        self.fusion_dim = self.validate_positive_int(fusion_dim, "fusion_dim")
        self.num_heads = self.validate_positive_int(num_heads, "num_heads")
        
        self.gpu_config = gpu_config or GPUConfig()
        
        # GPU availability and setup
        self.gpu_available = False
        self.device_type = None
        self.kernels = {}
        
        # Performance tracking
        self.gpu_performance_history = []
        self.memory_usage_history = []
        
        # Initialize GPU
        self._initialize_gpu()
        
        logger.info(f"GPUAcceleratedBioneuralFusion initialized: GPU={'available' if self.gpu_available else 'not available'}")
    
    def _initialize_gpu(self):
        """Initialize GPU acceleration"""
        
        if self.gpu_config.device_type == 'cuda' and CUPY_AVAILABLE:
            try:
                # Set CUDA device
                cp.cuda.Device(self.gpu_config.device_id).use()
                
                # Initialize memory pool if requested
                if self.gpu_config.memory_pool:
                    mempool = cp.get_default_memory_pool()
                    mempool.set_limit(fraction=0.8)  # Use 80% of GPU memory
                
                # Create CUDA kernels
                self.kernels['receptor'] = CUDAReceptorKernel(self.num_receptors, self.signal_dim)
                self.kernels['fusion'] = CUDAFusionKernel(self.signal_dim, self.fusion_dim, self.num_heads)
                
                self.gpu_available = True
                self.device_type = 'cuda'
                logger.info(f"CUDA acceleration initialized on device {self.gpu_config.device_id}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA: {e}")
                self._fallback_to_opencl()
        
        elif self.gpu_config.device_type == 'opencl' and OPENCL_AVAILABLE:
            self._initialize_opencl()
        
        else:
            logger.warning("No GPU acceleration available, falling back to CPU")
            self.gpu_available = False
    
    def _fallback_to_opencl(self):
        """Fallback to OpenCL if CUDA fails"""
        if OPENCL_AVAILABLE:
            self._initialize_opencl()
        else:
            logger.warning("OpenCL also not available, using CPU only")
            self.gpu_available = False
    
    def _initialize_opencl(self):
        """Initialize OpenCL acceleration"""
        try:
            self.kernels['receptor'] = OpenCLReceptorKernel(self.num_receptors, self.signal_dim)
            self.gpu_available = True
            self.device_type = 'opencl'
            logger.info("OpenCL acceleration initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenCL: {e}")
            self.gpu_available = False
    
    @robust_execution(recovery_strategy='cpu_fallback')
    def process_receptor_ensemble_gpu(self,
                                    signal: np.ndarray,
                                    sensitivity_profiles: np.ndarray,
                                    binding_affinities: np.ndarray,
                                    response_thresholds: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated receptor ensemble processing
        
        Args:
            signal: Input chemical signal
            sensitivity_profiles: Receptor sensitivity profiles
            binding_affinities: Receptor binding affinities
            response_thresholds: Receptor response thresholds
            
        Returns:
            Receptor activation vector
        """
        if not self.gpu_available or 'receptor' not in self.kernels:
            return self._process_receptor_ensemble_cpu(
                signal, sensitivity_profiles, binding_affinities, response_thresholds
            )
        
        start_time = time.time()
        
        try:
            # Execute GPU kernel
            activations = self.kernels['receptor'].execute(
                signal, sensitivity_profiles, binding_affinities, response_thresholds
            )
            
            # Record performance
            processing_time = time.time() - start_time
            self._record_gpu_performance('receptor', processing_time)
            
            return activations
            
        except Exception as e:
            logger.error(f"GPU receptor processing failed: {e}, falling back to CPU")
            return self._process_receptor_ensemble_cpu(
                signal, sensitivity_profiles, binding_affinities, response_thresholds
            )
    
    def _process_receptor_ensemble_cpu(self,
                                     signal: np.ndarray,
                                     sensitivity_profiles: np.ndarray,
                                     binding_affinities: np.ndarray,
                                     response_thresholds: np.ndarray) -> np.ndarray:
        """CPU fallback for receptor processing"""
        activations = np.zeros(self.num_receptors)
        
        for i in range(self.num_receptors):
            # Compute binding strength
            binding_strength = np.dot(signal, sensitivity_profiles[i])
            
            # Scale by affinity
            scaled_binding = binding_strength * binding_affinities[i]
            
            # Apply threshold and sigmoid
            if scaled_binding > response_thresholds[i]:
                x = 10.0 * (scaled_binding - response_thresholds[i])
                activations[i] = 1.0 / (1.0 + np.exp(-x))
        
        return activations
    
    @robust_execution(recovery_strategy='cpu_fallback')
    def process_neural_fusion_gpu(self,
                                input_tensor: np.ndarray,
                                query_weights: np.ndarray,
                                key_weights: np.ndarray,
                                value_weights: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated neural fusion processing
        
        Args:
            input_tensor: Input tensor for fusion
            query_weights: Query transformation weights
            key_weights: Key transformation weights  
            value_weights: Value transformation weights
            
        Returns:
            Fused representation
        """
        if not self.gpu_available or 'fusion' not in self.kernels:
            return self._process_neural_fusion_cpu(
                input_tensor, query_weights, key_weights, value_weights
            )
        
        start_time = time.time()
        
        try:
            # Execute GPU kernel
            fused_output = self.kernels['fusion'].execute(
                input_tensor, query_weights, key_weights, value_weights
            )
            
            # Record performance
            processing_time = time.time() - start_time
            self._record_gpu_performance('fusion', processing_time)
            
            return fused_output
            
        except Exception as e:
            logger.error(f"GPU fusion processing failed: {e}, falling back to CPU")
            return self._process_neural_fusion_cpu(
                input_tensor, query_weights, key_weights, value_weights
            )
    
    def _process_neural_fusion_cpu(self,
                                 input_tensor: np.ndarray,
                                 query_weights: np.ndarray,
                                 key_weights: np.ndarray,
                                 value_weights: np.ndarray) -> np.ndarray:
        """CPU fallback for neural fusion"""
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.reshape(1, -1)
        
        batch_size = input_tensor.shape[0]
        head_dim = self.fusion_dim // self.num_heads
        
        output = np.zeros((batch_size, self.fusion_dim))
        
        for head in range(self.num_heads):
            head_start = head * head_dim
            head_end = head_start + head_dim
            
            # Extract weights for this head
            q_weights = query_weights[:, head, :]
            k_weights = key_weights[:, head, :]
            v_weights = value_weights[:, head, :]
            
            # Compute attention
            Q = np.dot(input_tensor, q_weights)
            K = np.dot(input_tensor, k_weights)
            V = np.dot(input_tensor, v_weights)
            
            # Attention scores (simplified self-attention)
            scores = np.dot(Q, K.T) / np.sqrt(head_dim)
            attention_weights = 1.0 / (1.0 + np.exp(-scores))  # Sigmoid
            
            # Apply attention
            attended = np.dot(attention_weights, V)
            output[:, head_start:head_end] = attended
        
        return output
    
    def _record_gpu_performance(self, kernel_type: str, processing_time: float):
        """Record GPU performance metrics"""
        
        # Calculate FLOPS
        if kernel_type in self.kernels:
            flops = self.kernels[kernel_type].get_flops()
            flops_per_second = flops / processing_time if processing_time > 0 else 0
        else:
            flops_per_second = 0
        
        # Record metrics
        metrics = GPUPerformanceMetrics(
            kernel_execution_time=processing_time,
            memory_transfer_time=0.0,  # Would need more detailed profiling
            total_processing_time=processing_time,
            gpu_utilization=0.0,  # Would need GPU monitoring
            memory_bandwidth=0.0,  # Would need detailed profiling
            flops_per_second=flops_per_second
        )
        
        self.gpu_performance_history.append({
            'timestamp': time.time(),
            'kernel_type': kernel_type,
            'metrics': metrics
        })
        
        # Keep only recent history
        if len(self.gpu_performance_history) > 1000:
            self.gpu_performance_history = self.gpu_performance_history[-800:]
    
    def get_gpu_memory_info(self) -> GPUMemoryInfo:
        """Get current GPU memory information"""
        if not self.gpu_available:
            return GPUMemoryInfo(0, 0, 0, 0.0)
        
        if self.device_type == 'cuda' and CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                total_bytes = mempool.total_bytes()
                used_bytes = mempool.used_bytes()
                free_bytes = total_bytes - used_bytes
                
                return GPUMemoryInfo(
                    total_memory=total_bytes,
                    free_memory=free_bytes,
                    used_memory=used_bytes,
                    memory_utilization=used_bytes / max(total_bytes, 1)
                )
            except Exception as e:
                logger.error(f"Failed to get CUDA memory info: {e}")
        
        # Default/fallback
        return GPUMemoryInfo(0, 0, 0, 0.0)
    
    def benchmark_gpu_performance(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark GPU performance
        
        Args:
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance benchmark results
        """
        logger.info(f"Starting GPU performance benchmark: {num_iterations} iterations")
        
        # Generate test data
        test_signal = np.random.randn(self.signal_dim).astype(np.float32)
        test_profiles = np.random.randn(self.num_receptors, self.signal_dim).astype(np.float32)
        test_affinities = np.random.rand(self.num_receptors).astype(np.float32)
        test_thresholds = np.random.rand(self.num_receptors).astype(np.float32) * 0.5
        
        test_input = np.random.randn(self.signal_dim).astype(np.float32)
        test_q_weights = np.random.randn(self.signal_dim, self.num_heads, self.fusion_dim // self.num_heads).astype(np.float32)
        test_k_weights = np.random.randn(self.signal_dim, self.num_heads, self.fusion_dim // self.num_heads).astype(np.float32)
        test_v_weights = np.random.randn(self.signal_dim, self.num_heads, self.fusion_dim // self.num_heads).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            if 'receptor' in self.kernels:
                self.process_receptor_ensemble_gpu(test_signal, test_profiles, test_affinities, test_thresholds)
            if 'fusion' in self.kernels:
                self.process_neural_fusion_gpu(test_input, test_q_weights, test_k_weights, test_v_weights)
        
        # Benchmark receptor processing
        receptor_times = []
        fusion_times = []
        
        for i in range(num_iterations):
            # Receptor benchmark
            if 'receptor' in self.kernels:
                start_time = time.time()
                self.process_receptor_ensemble_gpu(test_signal, test_profiles, test_affinities, test_thresholds)
                receptor_times.append(time.time() - start_time)
            
            # Fusion benchmark
            if 'fusion' in self.kernels:
                start_time = time.time()
                self.process_neural_fusion_gpu(test_input, test_q_weights, test_k_weights, test_v_weights)
                fusion_times.append(time.time() - start_time)
        
        # Compute statistics
        benchmark_results = {
            'gpu_available': self.gpu_available,
            'device_type': self.device_type,
            'num_iterations': num_iterations,
            'receptor_processing': {
                'mean_time': np.mean(receptor_times) if receptor_times else 0.0,
                'std_time': np.std(receptor_times) if receptor_times else 0.0,
                'min_time': np.min(receptor_times) if receptor_times else 0.0,
                'max_time': np.max(receptor_times) if receptor_times else 0.0,
                'throughput': 1.0 / np.mean(receptor_times) if receptor_times else 0.0
            },
            'fusion_processing': {
                'mean_time': np.mean(fusion_times) if fusion_times else 0.0,
                'std_time': np.std(fusion_times) if fusion_times else 0.0,
                'min_time': np.min(fusion_times) if fusion_times else 0.0,
                'max_time': np.max(fusion_times) if fusion_times else 0.0,
                'throughput': 1.0 / np.mean(fusion_times) if fusion_times else 0.0
            }
        }
        
        # Add memory information
        memory_info = self.get_gpu_memory_info()
        benchmark_results['memory_info'] = {
            'total_memory_mb': memory_info.total_memory / (1024 * 1024),
            'used_memory_mb': memory_info.used_memory / (1024 * 1024),
            'free_memory_mb': memory_info.free_memory / (1024 * 1024),
            'memory_utilization': memory_info.memory_utilization
        }
        
        logger.info(f"GPU benchmark complete: receptor={benchmark_results['receptor_processing']['mean_time']:.6f}s, fusion={benchmark_results['fusion_processing']['mean_time']:.6f}s")
        
        return benchmark_results
    
    def optimize_memory_usage(self):
        """Optimize GPU memory usage"""
        if not self.gpu_available:
            return
        
        if self.device_type == 'cuda' and CUPY_AVAILABLE:
            # Clear memory pool
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            logger.info("CUDA memory pools cleared")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        # Analyze performance history
        if self.gpu_performance_history:
            recent_metrics = self.gpu_performance_history[-100:]  # Last 100 operations
            
            receptor_metrics = [m['metrics'] for m in recent_metrics if m['kernel_type'] == 'receptor']
            fusion_metrics = [m['metrics'] for m in recent_metrics if m['kernel_type'] == 'fusion']
            
            receptor_stats = {
                'count': len(receptor_metrics),
                'avg_time': np.mean([m.kernel_execution_time for m in receptor_metrics]) if receptor_metrics else 0.0,
                'avg_flops_per_sec': np.mean([m.flops_per_second for m in receptor_metrics]) if receptor_metrics else 0.0
            }
            
            fusion_stats = {
                'count': len(fusion_metrics),
                'avg_time': np.mean([m.kernel_execution_time for m in fusion_metrics]) if fusion_metrics else 0.0,
                'avg_flops_per_sec': np.mean([m.flops_per_second for m in fusion_metrics]) if fusion_metrics else 0.0
            }
        else:
            receptor_stats = {'count': 0, 'avg_time': 0.0, 'avg_flops_per_sec': 0.0}
            fusion_stats = {'count': 0, 'avg_time': 0.0, 'avg_flops_per_sec': 0.0}
        
        return {
            'gpu_acceleration': {
                'available': self.gpu_available,
                'device_type': self.device_type,
                'config': {
                    'device_id': self.gpu_config.device_id,
                    'precision': self.gpu_config.precision,
                    'memory_pool': self.gpu_config.memory_pool,
                    'batch_size': self.gpu_config.batch_size
                }
            },
            'performance_stats': {
                'receptor_processing': receptor_stats,
                'fusion_processing': fusion_stats,
                'total_operations': len(self.gpu_performance_history)
            },
            'memory_info': self.get_gpu_memory_info().__dict__,
            'capabilities': [
                'CUDA acceleration' if CUPY_AVAILABLE else 'CUDA not available',
                'OpenCL acceleration' if OPENCL_AVAILABLE else 'OpenCL not available',
                'Automatic CPU fallback',
                'Performance monitoring',
                'Memory optimization'
            ]
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get comprehensive GPU acceleration summary"""
        return {
            "accelerator_type": "GPUAcceleratedBioneuralFusion",
            "configuration": {
                "num_receptors": self.num_receptors,
                "signal_dim": self.signal_dim,
                "fusion_dim": self.fusion_dim,
                "num_heads": self.num_heads,
                "gpu_config": self.gpu_config.__dict__
            },
            "performance_summary": self.get_performance_summary(),
            "research_contributions": [
                "CUDA acceleration for receptor ensemble processing",
                "OpenCL cross-platform GPU support",
                "Automatic CPU fallback for robustness",
                "Real-time performance monitoring",
                "Memory-efficient GPU kernel implementations"
            ]
        }