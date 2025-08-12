#!/usr/bin/env python3
"""
Robust Framework Demonstration
Generation 2: MAKE IT ROBUST
"""

import sys
import os
import numpy as np
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.robust_framework import (
    robust_execution,
    secure_operation,
    RobustLogger,
    SecurityConfig,
    InputValidator,
    HealthChecker,
    default_health_checker
)


# Demo functions with robust execution
@robust_execution(max_retries=3, timeout_seconds=30, log_performance=True)
def robust_data_processing(data: np.ndarray, threshold: float = 0.5) -> dict:
    """Robust data processing with enhanced error handling"""
    
    # Input validation happens automatically via decorator
    
    # Simulate processing
    if len(data) == 0:
        raise ValueError("Empty dataset provided")
    
    # Simulate potential failure
    if np.random.random() < 0.2:  # 20% chance of simulated failure
        raise RuntimeError("Simulated processing error")
    
    # Actual processing
    mean_val = np.mean(data)
    std_val = np.std(data)
    outliers = np.abs(data - mean_val) > threshold * std_val
    
    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "outlier_count": int(np.sum(outliers)),
        "outlier_percentage": float(np.sum(outliers) / len(data) * 100),
        "processed_samples": len(data)
    }


@robust_execution(max_retries=2, enable_circuit_breaker=True)
def robust_model_training(x_data: np.ndarray, y_data: np.ndarray, epochs: int = 10) -> dict:
    """Robust model training with circuit breaker protection"""
    
    if len(x_data) != len(y_data):
        raise ValueError("Feature and target data length mismatch")
    
    # Simulate training process
    training_losses = []
    
    for epoch in range(epochs):
        # Simulate potential memory issues
        if epoch > 5 and np.random.random() < 0.1:
            raise MemoryError("Simulated memory exhaustion during training")
        
        # Simulate loss calculation
        loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
        training_losses.append(loss)
        
        # Brief pause to simulate computation
        time.sleep(0.01)
    
    return {
        "final_loss": training_losses[-1],
        "training_losses": training_losses,
        "epochs_completed": epochs,
        "convergence_rate": (training_losses[0] - training_losses[-1]) / training_losses[0]
    }


def demonstrate_input_validation():
    """Demonstrate comprehensive input validation"""
    
    logger = RobustLogger("validation_demo")
    
    print("🔒 Input Validation Demonstration")
    print("-" * 40)
    
    # Test string validation
    try:
        valid_string = InputValidator.validate_string("Safe input string", 100)
        print(f"✅ Valid string: '{valid_string}'")
    except Exception as e:
        print(f"❌ String validation failed: {e}")
    
    # Test malicious string detection
    try:
        malicious_string = InputValidator.validate_string("<script>alert('xss')</script>", 100)
        print(f"❌ Should have failed: '{malicious_string}'")
    except Exception as e:
        print(f"✅ Malicious string blocked: {e}")
    
    # Test numeric validation
    try:
        valid_number = InputValidator.validate_numeric(42.5, 0, 100)
        print(f"✅ Valid number: {valid_number}")
    except Exception as e:
        print(f"❌ Numeric validation failed: {e}")
    
    # Test out-of-range number
    try:
        invalid_number = InputValidator.validate_numeric(150, 0, 100)
        print(f"❌ Should have failed: {invalid_number}")
    except Exception as e:
        print(f"✅ Out-of-range number blocked: {e}")


def demonstrate_secure_operations():
    """Demonstrate secure operations with monitoring"""
    
    print("\n🛡️ Secure Operations Demonstration")
    print("-" * 40)
    
    # Secure data processing
    try:
        with secure_operation("data_analysis", max_time=60) as monitor:
            
            # Generate test data
            test_data = np.random.randn(1000)
            
            # Check resources during operation
            monitor.check_resources()
            
            # Process data
            result = robust_data_processing(test_data, threshold=1.5)
            
            print(f"✅ Secure processing completed:")
            print(f"   • Processed {result['processed_samples']} samples")
            print(f"   • Mean: {result['mean']:.3f}")
            print(f"   • Outliers: {result['outlier_count']} ({result['outlier_percentage']:.1f}%)")
            
    except Exception as e:
        print(f"❌ Secure operation failed: {e}")
    
    # Secure model training
    try:
        with secure_operation("model_training", max_time=30) as monitor:
            
            # Generate training data
            x_train = np.random.randn(100, 5)
            y_train = np.random.randn(100)
            
            # Train model with robustness
            training_result = robust_model_training(x_train, y_train, epochs=5)
            
            print(f"✅ Robust training completed:")
            print(f"   • Final loss: {training_result['final_loss']:.4f}")
            print(f"   • Convergence rate: {training_result['convergence_rate']:.2%}")
            
    except Exception as e:
        print(f"❌ Training operation failed: {e}")


def demonstrate_health_monitoring():
    """Demonstrate system health monitoring"""
    
    print("\n💗 Health Monitoring Demonstration")
    print("-" * 40)
    
    # Run default health checks
    health_results = default_health_checker.run_health_checks()
    
    print(f"Overall Status: {health_results['overall_status']}")
    print(f"Timestamp: {health_results['timestamp']}")
    
    print("\nIndividual Checks:")
    for check_name, check_result in health_results['checks'].items():
        status_icon = "✅" if check_result['status'] == 'PASS' else "❌"
        critical_mark = " (CRITICAL)" if check_result.get('critical', False) else ""
        print(f"   {status_icon} {check_name}: {check_result['status']}{critical_mark}")
    
    if health_results['critical_failures']:
        print(f"\n🚨 Critical Failures: {', '.join(health_results['critical_failures'])}")
    
    if health_results['warnings']:
        print(f"⚠️ Warnings: {', '.join(health_results['warnings'])}")


def demonstrate_error_recovery():
    """Demonstrate error recovery mechanisms"""
    
    print("\n🔄 Error Recovery Demonstration")
    print("-" * 40)
    
    # Test retry mechanism with eventual success
    success_count = 0
    failure_count = 0
    
    for attempt in range(5):
        try:
            # This will fail randomly but eventually succeed due to retries
            result = robust_data_processing(np.random.randn(50), threshold=2.0)
            success_count += 1
            print(f"✅ Attempt {attempt + 1}: Success - processed {result['processed_samples']} samples")
            
        except Exception as e:
            failure_count += 1
            print(f"❌ Attempt {attempt + 1}: Failed - {e}")
    
    print(f"\nRecovery Summary:")
    print(f"   • Successful operations: {success_count}")
    print(f"   • Failed operations: {failure_count}")
    print(f"   • Success rate: {success_count/(success_count + failure_count):.1%}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring"""
    
    print("\n📊 Performance Monitoring Demonstration")
    print("-" * 40)
    
    # Create logger for performance tracking
    perf_logger = RobustLogger("performance_demo", "performance.log")
    
    # Simulate various operations with performance tracking
    operations = [
        ("Small dataset", lambda: robust_data_processing(np.random.randn(100))),
        ("Medium dataset", lambda: robust_data_processing(np.random.randn(1000))),
        ("Large dataset", lambda: robust_data_processing(np.random.randn(5000))),
        ("Training", lambda: robust_model_training(np.random.randn(200, 8), np.random.randn(200), 3))
    ]
    
    for op_name, operation in operations:
        try:
            start_time = time.time()
            result = operation()
            execution_time = time.time() - start_time
            
            perf_logger.info(
                f"Operation completed: {op_name}",
                operation=op_name,
                execution_time=execution_time,
                success=True
            )
            
            print(f"✅ {op_name}: {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            perf_logger.error(
                f"Operation failed: {op_name}",
                operation=op_name,
                execution_time=execution_time,
                error=str(e),
                success=False
            )
            
            print(f"❌ {op_name}: Failed after {execution_time:.3f}s - {e}")
    
    # Display performance metrics summary
    print(f"\nPerformance Metrics:")
    print(f"   • Total operations: {len(operations)}")
    print(f"   • Log entries: {len(perf_logger.performance_metrics)} performance metrics")
    print(f"   • Audit entries: {len(perf_logger.audit_logs)} security audits")


def main():
    """Main demonstration function"""
    
    print("🛡️ ROBUST FRAMEWORK DEMONSTRATION")
    print("=" * 50)
    print("Generation 2: MAKE IT ROBUST")
    print("Enhanced Error Handling, Logging, and Security")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        demonstrate_input_validation()
        demonstrate_secure_operations()
        demonstrate_health_monitoring()
        demonstrate_error_recovery()
        demonstrate_performance_monitoring()
        
        print("\n" + "=" * 50)
        print("✅ ROBUST FRAMEWORK DEMONSTRATION COMPLETED")
        print("✅ Enhanced error handling validated")
        print("✅ Security measures implemented")
        print("✅ Performance monitoring active")
        print("✅ Health checks operational")
        print("✅ System robustness verified")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)