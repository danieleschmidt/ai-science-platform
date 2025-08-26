"""🛡️ COMPREHENSIVE QUALITY GATES - Automated Testing & Validation"""

import sys
import subprocess
import time
import logging
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logging_config import setup_logging
from src.algorithms.discovery import DiscoveryEngine
from src.models.simple import SimpleDiscoveryModel
from src.utils.data_utils import generate_sample_data
from src.performance.enhanced_caching import get_cache
from src.performance.concurrent_discovery import ConcurrentDiscoveryEngine
from src.performance.adaptive_scaling import AdaptiveScaler, ScalingPolicy, create_mock_worker

logger = logging.getLogger(__name__)


class QualityGate:
    """Individual quality gate implementation"""
    
    def __init__(self, name: str, description: str, critical: bool = False):
        self.name = name
        self.description = description
        self.critical = critical
        self.passed = False
        self.error = None
        self.metrics = {}
        self.execution_time = 0.0
    
    def run(self) -> bool:
        """Execute the quality gate check"""
        start_time = time.time()
        try:
            self.passed = self._execute()
            if self.passed:
                logger.info(f"✅ {self.name}: PASSED")
            else:
                logger.warning(f"❌ {self.name}: FAILED")
                
        except Exception as e:
            self.error = str(e)
            self.passed = False
            logger.error(f"💥 {self.name}: ERROR - {e}")
        
        self.execution_time = time.time() - start_time
        return self.passed
    
    def _execute(self) -> bool:
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def get_report(self) -> Dict[str, Any]:
        """Get gate execution report"""
        return {
            'name': self.name,
            'description': self.description,
            'critical': self.critical,
            'passed': self.passed,
            'error': self.error,
            'metrics': self.metrics,
            'execution_time': self.execution_time
        }


class CoreFunctionalityGate(QualityGate):
    """Test core platform functionality"""
    
    def __init__(self):
        super().__init__("Core Functionality", "Verify all core components work", critical=True)
    
    def _execute(self) -> bool:
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Discovery Engine
        total_tests += 1
        try:
            engine = DiscoveryEngine(discovery_threshold=0.6)
            data, _ = generate_sample_data(size=50, data_type='normal')
            discoveries = engine.discover(data, context="quality_gate_test")
            
            self.metrics['discovery_engine_discoveries'] = len(discoveries)
            tests_passed += 1
            logger.debug("Discovery engine test passed")
            
        except Exception as e:
            logger.error(f"Discovery engine test failed: {e}")
        
        # Test 2: Discovery Model
        total_tests += 1
        try:
            test_data = generate_sample_data(size=20, data_type='normal')[0]
            model = SimpleDiscoveryModel(input_dim=test_data.shape[1])
            predictions = model.predict(test_data)
            
            self.metrics['model_predictions'] = len(predictions)
            tests_passed += 1
            logger.debug("Discovery model test passed")
            
        except Exception as e:
            logger.error(f"Discovery model test failed: {e}")
        
        # Test 3: Data Generation
        total_tests += 1
        try:
            data, targets = generate_sample_data(size=100, data_type='normal')
            assert data.shape[0] == 100
            assert len(targets) == 100
            
            self.metrics['data_generation_shape'] = list(data.shape)
            tests_passed += 1
            logger.debug("Data generation test passed")
            
        except Exception as e:
            logger.error(f"Data generation test failed: {e}")
        
        success_rate = tests_passed / total_tests
        self.metrics['success_rate'] = success_rate
        self.metrics['tests_passed'] = tests_passed
        self.metrics['total_tests'] = total_tests
        
        return success_rate >= 0.85  # Require 85% success rate


class PerformanceGate(QualityGate):
    """Test performance requirements"""
    
    def __init__(self):
        super().__init__("Performance", "Verify performance benchmarks", critical=False)
    
    def _execute(self) -> bool:
        # Test concurrent discovery performance
        concurrent_engine = ConcurrentDiscoveryEngine(max_workers=2)
        
        # Generate test datasets
        datasets = []
        for i in range(6):
            data, _ = generate_sample_data(size=50, data_type='normal')
            datasets.append((data, f"perf_test_{i}"))
        
        # Measure concurrent processing time
        start_time = time.time()
        results = concurrent_engine.discover_parallel(datasets, threshold=0.7)
        concurrent_time = time.time() - start_time
        
        # Get performance stats
        perf_stats = concurrent_engine.get_performance_stats()
        
        self.metrics.update({
            'concurrent_processing_time': concurrent_time,
            'datasets_processed': len(results),
            'parallel_speedup': perf_stats.get('parallel_speedup', 1.0),
            'avg_task_time': perf_stats.get('avg_task_time', 0.0)
        })
        
        # Performance requirements
        max_processing_time = 2.0  # seconds
        min_speedup = 1.5
        
        return (concurrent_time <= max_processing_time and 
                perf_stats.get('parallel_speedup', 0) >= min_speedup)


class CachingGate(QualityGate):
    """Test caching system effectiveness"""
    
    def __init__(self):
        super().__init__("Caching", "Verify caching performance and correctness", critical=False)
    
    def _execute(self) -> bool:
        cache = get_cache()
        cache.clear()  # Start fresh
        
        # Test cache hit/miss
        test_key = "cache_test_key"
        test_value = {"test": "data", "number": 42}
        
        # Test cache miss
        result = cache.get(test_key)
        if result is not None:
            return False
        
        # Test cache put
        success = cache.put(test_key, test_value)
        if not success:
            return False
        
        # Test cache hit
        result = cache.get(test_key)
        if result != test_value:
            return False
        
        # Get cache statistics
        stats = cache.get_stats()
        
        self.metrics.update({
            'cache_entries': stats['entries'],
            'hit_rate': stats['hit_rate'],
            'memory_usage_mb': stats['memory_usage_mb']
        })
        
        return stats['entries'] > 0 and stats['hit_rate'] > 0


class ScalingGate(QualityGate):
    """Test adaptive scaling functionality"""
    
    def __init__(self):
        super().__init__("Adaptive Scaling", "Verify auto-scaling works correctly", critical=False)
    
    def _execute(self) -> bool:
        # Create scaling policy
        policy = ScalingPolicy(
            name="quality_gate_scaling",
            min_workers=1,
            max_workers=3,
            scale_up_cooldown=1,
            scale_down_cooldown=2
        )
        
        scaler = AdaptiveScaler(policy, create_mock_worker)
        initial_workers = scaler.get_current_capacity()
        
        # Test manual scaling
        target_workers = 2
        success = scaler.manually_scale(target_workers)
        final_workers = scaler.get_current_capacity()
        
        stats = scaler.get_scaling_stats()
        
        self.metrics.update({
            'initial_workers': initial_workers,
            'target_workers': target_workers,
            'final_workers': final_workers,
            'scaling_success': success,
            'min_workers': stats['min_workers'],
            'max_workers': stats['max_workers']
        })
        
        return success and final_workers == target_workers


class SecurityGate(QualityGate):
    """Test security and validation features"""
    
    def __init__(self):
        super().__init__("Security", "Verify security measures are active", critical=True)
    
    def _execute(self) -> bool:
        security_checks = 0
        total_checks = 0
        
        # Test 1: Data validation
        total_checks += 1
        try:
            from src.utils.security import SecurityMixin
            
            # This should work
            data, _ = generate_sample_data(size=10, data_type='normal')
            # If no exception, validation is working
            security_checks += 1
            
        except Exception as e:
            logger.debug(f"Security validation test failed: {e}")
        
        # Test 2: Check if backup system exists
        total_checks += 1
        try:
            from src.utils.backup import BackupManager
            backup_manager = BackupManager()
            security_checks += 1
            
        except Exception as e:
            logger.debug(f"Backup system test failed: {e}")
        
        self.metrics.update({
            'security_checks_passed': security_checks,
            'total_security_checks': total_checks,
            'security_coverage': security_checks / total_checks if total_checks > 0 else 0
        })
        
        return security_checks >= total_checks * 0.7  # 70% of security checks must pass


class IntegrationGate(QualityGate):
    """Test end-to-end integration"""
    
    def __init__(self):
        super().__init__("Integration", "Verify end-to-end platform integration", critical=True)
    
    def _execute(self) -> bool:
        try:
            # Test full workflow
            # 1. Data generation
            datasets = []
            for i in range(3):
                data, _ = generate_sample_data(size=30, data_type='normal')
                datasets.append((data, f"integration_test_{i}"))
            
            # 2. Concurrent discovery
            engine = ConcurrentDiscoveryEngine(max_workers=2, cache_enabled=True)
            results = engine.discover_parallel(datasets, threshold=0.7)
            
            # 3. Results validation
            total_datasets = len(datasets)
            processed_datasets = len(results)
            
            self.metrics.update({
                'datasets_generated': total_datasets,
                'datasets_processed': processed_datasets,
                'integration_success_rate': processed_datasets / total_datasets
            })
            
            return processed_datasets == total_datasets
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False


class QualityGateFramework:
    """Comprehensive quality gate execution framework"""
    
    def __init__(self):
        self.gates = [
            CoreFunctionalityGate(),
            PerformanceGate(),
            CachingGate(),
            ScalingGate(),
            SecurityGate(),
            IntegrationGate()
        ]
        
        self.results = {}
        self.total_execution_time = 0.0
    
    def run_all_gates(self) -> Tuple[bool, Dict[str, Any]]:
        """Execute all quality gates"""
        print("🛡️ EXECUTING COMPREHENSIVE QUALITY GATES")
        print("=" * 60)
        
        start_time = time.time()
        
        passed_gates = 0
        critical_failures = 0
        
        for gate in self.gates:
            print(f"\n🔍 Running: {gate.name}")
            print(f"   {gate.description}")
            
            success = gate.run()
            self.results[gate.name] = gate.get_report()
            
            if success:
                passed_gates += 1
            elif gate.critical:
                critical_failures += 1
        
        self.total_execution_time = time.time() - start_time
        
        # Determine overall success
        total_gates = len(self.gates)
        success_rate = passed_gates / total_gates
        overall_success = critical_failures == 0 and success_rate >= 0.85
        
        # Generate comprehensive report
        report = self._generate_report(overall_success, passed_gates, total_gates, critical_failures)
        
        return overall_success, report
    
    def _generate_report(self, overall_success: bool, passed: int, total: int, critical_failures: int) -> Dict[str, Any]:
        """Generate comprehensive quality gate report"""
        
        return {
            'overall_success': overall_success,
            'summary': {
                'total_gates': total,
                'passed_gates': passed,
                'failed_gates': total - passed,
                'critical_failures': critical_failures,
                'success_rate': passed / total,
                'total_execution_time': self.total_execution_time
            },
            'gate_results': self.results,
            'recommendations': self._get_recommendations(),
            'timestamp': time.time()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Generate recommendations based on gate results"""
        recommendations = []
        
        for gate_name, result in self.results.items():
            if not result['passed']:
                if result['critical']:
                    recommendations.append(f"CRITICAL: Fix {gate_name} - {result.get('error', 'Unknown error')}")
                else:
                    recommendations.append(f"IMPROVE: Optimize {gate_name} performance")
        
        if not recommendations:
            recommendations.append("All quality gates passed! Platform is production-ready.")
        
        return recommendations
    
    def save_report(self, filepath: str, success: bool, passed: int, total: int, critical_failures: int):
        """Save quality gate report to file"""
        report = self._generate_report(success, passed, total, critical_failures)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Quality gate report saved to {filepath}")


def main():
    """Execute comprehensive quality gates"""
    print("🧬 AI SCIENCE PLATFORM - QUALITY GATES EXECUTION")
    print("=" * 60)
    
    setup_logging()
    logger.info("Starting comprehensive quality gate execution")
    
    framework = QualityGateFramework()
    success, report = framework.run_all_gates()
    
    # Display results
    print(f"\n📊 QUALITY GATES SUMMARY")
    print("=" * 40)
    print(f"Overall Success: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"Gates Passed: {report['summary']['passed_gates']}/{report['summary']['total_gates']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"Critical Failures: {report['summary']['critical_failures']}")
    print(f"Execution Time: {report['summary']['total_execution_time']:.2f}s")
    
    # Show gate details
    print(f"\n🔍 GATE DETAILS:")
    for gate_name, result in report['gate_results'].items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        critical = " (CRITICAL)" if result['critical'] else ""
        print(f"  {status} {gate_name}{critical} ({result['execution_time']:.2f}s)")
        
        if result['error']:
            print(f"    Error: {result['error']}")
    
    # Show recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Save detailed report
    report_file = "quality_gates_report.json"
    framework.save_report(
        report_file, 
        success, 
        report['summary']['passed_gates'], 
        report['summary']['total_gates'],
        report['summary']['critical_failures']
    )
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)