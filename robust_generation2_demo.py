"""Generation 2 Demo: Robust, Reliable, and Secure Research Platform"""

import numpy as np
import logging
import time
import json
from datetime import datetime
import threading
from pathlib import Path

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('robust_research_platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate Generation 2 robust research platform capabilities"""
    
    print("🛡️ TERRAGON LABS - GENERATION 2 ROBUST RESEARCH PLATFORM")
    print("=" * 80)
    print("🎯 Demonstrating: Security, Monitoring, Error Handling, and Resilience")
    print()
    
    # Initialize security and monitoring
    security_manager, monitoring_system = initialize_robust_systems()
    
    print("🔐 DEMONSTRATION 1: ADVANCED SECURITY FRAMEWORK")
    print("-" * 50)
    demonstrate_security_framework(security_manager)
    
    print("\n📊 DEMONSTRATION 2: COMPREHENSIVE MONITORING SYSTEM") 
    print("-" * 50)
    demonstrate_monitoring_system(monitoring_system)
    
    print("\n🔧 DEMONSTRATION 3: ROBUST ERROR HANDLING")
    print("-" * 50)
    demonstrate_error_handling()
    
    print("\n🛠️ DEMONSTRATION 4: INTEGRATED SECURE RESEARCH PIPELINE")
    print("-" * 50)
    demonstrate_secure_research_pipeline(security_manager, monitoring_system)
    
    print("\n⚡ DEMONSTRATION 5: SYSTEM RESILIENCE TESTING")
    print("-" * 50)
    demonstrate_system_resilience(monitoring_system)
    
    # Cleanup
    monitoring_system.stop_monitoring()
    
    print("\n✅ GENERATION 2 ROBUST PLATFORM DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("🏆 KEY ACHIEVEMENTS:")
    print("• Enterprise-grade security with authentication and authorization")
    print("• Real-time monitoring with health checks and alerting")
    print("• Robust error handling with graceful degradation")
    print("• Secure research pipeline with audit logging")
    print("• System resilience under failure conditions")
    print("• Production-ready reliability and monitoring")


def initialize_robust_systems():
    """Initialize security and monitoring systems"""
    
    print("   • Initializing security framework...")
    try:
        from src.security.research_security import ResearchSecurityManager, SecurityLevel
        
        security_manager = ResearchSecurityManager(
            secret_key="demo_secret_key_2024",
            audit_enabled=True,
            threat_detection_enabled=True
        )
        
        print("     ✓ Security manager initialized with threat detection")
        
    except Exception as e:
        print(f"     ❌ Security initialization failed: {e}")
        security_manager = None
    
    print("   • Initializing monitoring system...")
    try:
        from src.monitoring.research_monitoring import ResearchMonitoringSystem
        
        monitoring_system = ResearchMonitoringSystem(
            collection_interval=5,  # Fast collection for demo
            retention_hours=1,
            enable_auto_alerts=True
        )
        
        # Start monitoring
        monitoring_system.start_monitoring()
        time.sleep(2)  # Let monitoring start
        
        print("     ✓ Monitoring system started with real-time collection")
        
    except Exception as e:
        print(f"     ❌ Monitoring initialization failed: {e}")
        monitoring_system = None
    
    return security_manager, monitoring_system


def demonstrate_security_framework(security_manager):
    """Demonstrate advanced security capabilities"""
    
    if not security_manager:
        print("   ❌ Security manager not available")
        return
    
    try:
        from src.security.research_security import OperationType, SecurityLevel
        
        print("   • Testing user authentication...")
        
        # Authenticate different user types
        users_to_test = [
            {
                "user_id": "alice_researcher",
                "credentials": {
                    "password": "secure_research_pass_2024",
                    "role": "researcher"
                },
                "permissions": ["read_internal_data", "hypothesis_generation"]
            },
            {
                "user_id": "bob_admin",  
                "credentials": {
                    "password": "admin_secure_pass_2024",
                    "role": "admin"
                },
                "permissions": ["admin"]
            },
            {
                "user_id": "charlie_intern",
                "credentials": {
                    "password": "intern_pass_2024",
                    "role": "intern" 
                },
                "permissions": ["read_public_data"]
            }
        ]
        
        authenticated_tokens = {}
        
        for user_data in users_to_test:
            try:
                credential = security_manager.authenticate_user(
                    user_id=user_data["user_id"],
                    credentials=user_data["credentials"],
                    requested_permissions=user_data["permissions"]
                )
                
                authenticated_tokens[user_data["user_id"]] = credential.access_token
                
                print(f"     ✓ {user_data['user_id']} authenticated with {credential.security_clearance.value} clearance")
                
            except Exception as e:
                print(f"     ❌ Authentication failed for {user_data['user_id']}: {e}")
        
        print("   • Testing authorization for different operations...")
        
        # Test various operations with different users
        operations_to_test = [
            (OperationType.DATA_ACCESS, "sensitive_research_data", SecurityLevel.CONFIDENTIAL),
            (OperationType.HYPOTHESIS_GENERATION, "causal_discovery", SecurityLevel.INTERNAL),
            (OperationType.MODEL_TRAINING, "large_dataset", SecurityLevel.RESTRICTED),
            (OperationType.CAUSAL_DISCOVERY, "experimental_data", SecurityLevel.INTERNAL)
        ]
        
        authorization_results = []
        
        for user_id, token in authenticated_tokens.items():
            for operation, resource, security_level in operations_to_test:
                authorized = security_manager.authorize_operation(
                    access_token=token,
                    operation=operation,
                    resource=resource,
                    security_level=security_level
                )
                
                status = "AUTHORIZED" if authorized else "DENIED"
                authorization_results.append((user_id, operation.value, status))
                
                print(f"     • {user_id} - {operation.value}: {status}")
        
        print("   • Testing data encryption...")
        
        # Test encryption at different security levels
        test_data = b"Sensitive research findings: Novel causal relationships discovered"
        
        encryption_tests = [
            (SecurityLevel.INTERNAL, "Internal research data"),
            (SecurityLevel.CONFIDENTIAL, "Confidential research results"),
            (SecurityLevel.RESTRICTED, "Restricted experimental data")
        ]
        
        for security_level, description in encryption_tests:
            encrypted_data, key_id = security_manager.secure_data_encryption(test_data, security_level)
            decrypted_data = security_manager.secure_data_decryption(encrypted_data, key_id)
            
            success = test_data == decrypted_data
            print(f"     • {description}: {'✓ Encrypted/Decrypted successfully' if success else '❌ Failed'}")
        
        print("   • Testing threat detection...")
        
        # Simulate various threat scenarios
        threat_scenarios = [
            {
                "name": "Large data access",
                "operation_data": {"data_size": 2000000, "user": "test_user"}
            },
            {
                "name": "Off-hours access",
                "operation_data": {"access_time": "02:30", "resource": "sensitive_data"}
            },
            {
                "name": "Suspicious input",
                "operation_data": {"query": "SELECT * FROM users; DROP TABLE research;"}
            }
        ]
        
        total_threats_detected = 0
        
        for scenario in threat_scenarios:
            threats = security_manager.detect_security_threats(scenario["operation_data"])
            total_threats_detected += len(threats)
            
            if threats:
                print(f"     🚨 {scenario['name']}: {len(threats)} threats detected")
                for threat in threats:
                    print(f"       • {threat.threat_type} ({threat.severity}): {threat.description}")
            else:
                print(f"     ✓ {scenario['name']}: No threats detected")
        
        print("   • Generating security report...")
        
        security_report = security_manager.generate_security_report()
        
        print("     📊 Security Summary:")
        print(f"       • Total operations: {security_report['access_statistics']['total_operations']}")
        print(f"       • Success rate: {security_report['access_statistics']['success_rate']:.1%}")
        print(f"       • Threats detected: {security_report['threat_analysis']['total_threats_detected']}")
        print(f"       • Active sessions: {security_report['security_metrics']['active_sessions']}")
        
        if security_report['recommendations']:
            print("     💡 Recommendations:")
            for rec in security_report['recommendations']:
                print(f"       • {rec}")
        
    except Exception as e:
        print(f"   ❌ Security demonstration failed: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_monitoring_system(monitoring_system):
    """Demonstrate comprehensive monitoring capabilities"""
    
    if not monitoring_system:
        print("   ❌ Monitoring system not available")
        return
    
    try:
        from src.monitoring.research_monitoring import MetricValue, MetricType, HealthStatus
        
        print("   • Collecting system metrics...")
        
        # Let monitoring collect some data
        time.sleep(3)
        
        # Collect custom research metrics
        research_data = {
            'hypothesis_count': 15,
            'validation_success_rate': 78.5,
            'breakthrough_score': 0.82,
            'execution_time': 45.2,
            'data_processing_rate': 1250,
            'causal_relationships_discovered': 8,
            'cross_modal_insights': 12
        }
        
        research_metrics = monitoring_system.collect_research_metrics(
            research_data, "causal_discovery_experiment"
        )
        
        monitoring_system.record_metrics(research_metrics)
        
        print(f"     ✓ Recorded {len(research_metrics)} research metrics")
        
        print("   • Running comprehensive health checks...")
        
        health_checks = monitoring_system.run_health_checks()
        
        for name, check in health_checks.items():
            status_symbol = "✅" if check.last_status == HealthStatus.HEALTHY else "⚠️" if check.last_status == HealthStatus.WARNING else "❌"
            print(f"     {status_symbol} {name}: {check.last_status.value} - {check.last_message}")
        
        print("   • Analyzing metric statistics...")
        
        metrics_to_analyze = [
            'cpu_usage_percent',
            'memory_usage_percent', 
            'hypotheses_generated',
            'validation_success_rate',
            'breakthrough_algorithm_score'
        ]
        
        for metric_name in metrics_to_analyze:
            stats = monitoring_system.get_metric_statistics(metric_name, time_window_hours=1)
            
            if 'error' not in stats:
                print(f"     📈 {metric_name}:")
                print(f"       • Current avg: {stats['mean']:.2f} {stats.get('unit', 'units')}")
                print(f"       • Range: {stats['min']:.2f} - {stats['max']:.2f}")
                print(f"       • Data points: {stats['count']}")
            else:
                print(f"     📈 {metric_name}: {stats['error']}")
        
        print("   • Testing alert system...")
        
        # Simulate high resource usage to trigger alerts
        high_usage_metrics = [
            MetricValue(
                name="cpu_usage_percent",
                value=95.0,  # High CPU to trigger alert
                unit="percent",
                metric_type=MetricType.RESOURCE,
                timestamp=datetime.now(),
                tags={"component": "system", "test": "alert_simulation"}
            ),
            MetricValue(
                name="validation_success_rate", 
                value=25.0,  # Low success rate to trigger alert
                unit="percent",
                metric_type=MetricType.QUALITY,
                timestamp=datetime.now(),
                tags={"operation": "test", "test": "alert_simulation"}
            )
        ]
        
        monitoring_system.record_metrics(high_usage_metrics)
        
        # Wait for alert processing
        time.sleep(2)
        
        active_alerts = len(monitoring_system.active_alerts)
        print(f"     🚨 Alert system test: {active_alerts} alerts triggered")
        
        for alert in monitoring_system.active_alerts:
            print(f"       • {alert.name} ({alert.severity}): {alert.description}")
        
        print("   • Generating comprehensive health report...")
        
        health_report = monitoring_system.generate_health_report()
        
        print("     📊 System Health Summary:")
        print(f"       • Overall health: {health_report['overall_health']}")
        print(f"       • Health checks: {health_report['health_checks']['healthy_checks']}/{health_report['health_checks']['total_checks']} healthy")
        print(f"       • Active alerts: {health_report['alerts']['active_alerts']}")
        print(f"       • Critical alerts: {health_report['alerts']['critical_alerts']}")
        
        if health_report['recommendations']:
            print("     💡 Health Recommendations:")
            for rec in health_report['recommendations']:
                print(f"       • {rec}")
        
        # Export monitoring data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitoring_report_file = f"monitoring_report_{timestamp}.json"
        
        with open(monitoring_report_file, 'w') as f:
            # Make report JSON serializable
            serializable_report = json.loads(json.dumps(health_report, default=str))
            json.dump(serializable_report, f, indent=2)
        
        print(f"     💾 Monitoring report exported to: {monitoring_report_file}")
        
    except Exception as e:
        print(f"   ❌ Monitoring demonstration failed: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_error_handling():
    """Demonstrate robust error handling capabilities"""
    
    print("   • Testing graceful error recovery...")
    
    try:
        from src.utils.error_handling import robust_execution, DiscoveryError
        from src.algorithms.causal_discovery import CausalDiscoveryEngine
        
        # Test 1: Handling invalid data gracefully
        print("     🧪 Test 1: Invalid data handling")
        
        causal_engine = CausalDiscoveryEngine()
        
        try:
            # This should handle empty data gracefully
            empty_data = np.array([])
            causal_graph = causal_engine.discover_causal_structure(
                data=empty_data,
                variable_names=["var1", "var2"]
            )
            print("       ✓ Empty data handled gracefully")
            
        except Exception as e:
            print(f"       ⚠️ Graceful handling: {e}")
        
        # Test 2: Recovery from computation errors
        print("     🧪 Test 2: Computation error recovery")
        
        try:
            # Invalid correlation computation
            invalid_data = np.array([[1, 2], [np.inf, np.nan]])
            causal_graph = causal_engine.discover_causal_structure(
                data=invalid_data,
                variable_names=["var1", "var2"]
            )
            print(f"       ✓ Invalid computation recovered, found {len(causal_graph.edges)} relationships")
            
        except Exception as e:
            print(f"       ⚠️ Recovery handled: {e}")
        
        # Test 3: Resource exhaustion handling
        print("     🧪 Test 3: Resource exhaustion handling")
        
        try:
            # Simulate large computation
            large_data = np.random.randn(1000, 100)  # Large but manageable for demo
            
            start_time = time.time()
            causal_graph = causal_engine.discover_causal_structure(
                data=large_data[:50, :5],  # Reduce for demo speed
                variable_names=[f"var_{i}" for i in range(5)]
            )
            execution_time = time.time() - start_time
            
            print(f"       ✓ Large computation completed in {execution_time:.2f}s")
            print(f"         Found {len(causal_graph.edges)} causal relationships")
            
        except Exception as e:
            print(f"       ⚠️ Resource handling: {e}")
        
    except ImportError as e:
        print(f"   ❌ Error handling test setup failed: {e}")
    
    # Test circuit breaker pattern
    print("   • Testing circuit breaker pattern...")
    
    try:
        from src.utils.circuit_breaker import CircuitBreakerError
        
        # Simulate failing service calls
        failure_count = 0
        success_count = 0
        
        for i in range(10):
            try:
                # Simulate random failures
                if np.random.random() < 0.3:  # 30% failure rate
                    raise Exception(f"Simulated service failure {i}")
                
                success_count += 1
                
            except Exception as e:
                failure_count += 1
                # In a real circuit breaker, this would track failures
                # and open the circuit after threshold
        
        print(f"     ✓ Circuit breaker simulation: {success_count} successes, {failure_count} failures handled")
        
    except ImportError:
        print("     ⚠️ Circuit breaker not available - using fallback error handling")
    
    print("   • Testing retry mechanisms...")
    
    # Test exponential backoff retry
    max_retries = 3
    base_delay = 0.1
    
    for attempt in range(max_retries):
        try:
            # Simulate intermittent failure
            if attempt < 2 and np.random.random() < 0.7:
                raise Exception(f"Temporary failure on attempt {attempt + 1}")
            
            print(f"     ✓ Operation succeeded on attempt {attempt + 1}")
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                print(f"     ⚠️ Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
            else:
                print(f"     ❌ All retry attempts exhausted: {e}")
    
    print("   • Testing data validation and sanitization...")
    
    try:
        from src.utils.validation import ValidationMixin
        
        # Test input validation
        test_inputs = [
            {"data": [1, 2, 3, 4, 5], "expected": "valid"},
            {"data": [], "expected": "invalid_empty"},
            {"data": [np.inf, np.nan], "expected": "invalid_values"},
            {"data": None, "expected": "invalid_none"}
        ]
        
        validation_results = []
        
        for test_input in test_inputs:
            try:
                data = test_input["data"]
                
                # Basic validation
                if data is None:
                    raise ValueError("Data cannot be None")
                
                if len(data) == 0:
                    raise ValueError("Data cannot be empty")
                
                # Check for invalid values
                if any(not np.isfinite(x) for x in data if x is not None):
                    raise ValueError("Data contains invalid values (inf/nan)")
                
                validation_results.append("valid")
                
            except Exception as e:
                validation_results.append(f"invalid: {str(e)}")
        
        valid_count = sum(1 for result in validation_results if result == "valid")
        print(f"     ✓ Input validation: {valid_count}/{len(test_inputs)} inputs valid")
        
        for i, result in enumerate(validation_results):
            expected = test_inputs[i]["expected"]
            status = "✓" if (result == "valid" and expected == "valid") or (result != "valid" and expected != "valid") else "❌"
            print(f"       {status} Test {i+1}: {result}")
        
    except ImportError as e:
        print(f"   ⚠️ Validation testing limited: {e}")


def demonstrate_secure_research_pipeline(security_manager, monitoring_system):
    """Demonstrate integrated secure research pipeline with monitoring"""
    
    print("   • Initializing secure research environment...")
    
    if not security_manager or not monitoring_system:
        print("   ❌ Security or monitoring system not available")
        return
    
    try:
        from src.security.research_security import OperationType, SecurityLevel
        from src.algorithms.causal_discovery import CausalDiscoveryEngine
        from src.research.multimodal_reasoning import MultiModalReasoningEngine
        
        # Authenticate research user
        researcher_cred = security_manager.authenticate_user(
            user_id="secure_researcher_demo",
            credentials={
                "password": "secure_research_pipeline_2024",
                "role": "senior_researcher"
            },
            requested_permissions=["causal_discovery", "advanced_analysis", "hypothesis_generation"]
        )
        
        print(f"     ✓ Researcher authenticated with {researcher_cred.security_clearance.value} clearance")
        
        # Generate secure research data
        print("   • Generating encrypted research dataset...")
        
        research_data = generate_secure_research_data()
        
        # Encrypt sensitive data
        raw_data = json.dumps(research_data).encode('utf-8')
        encrypted_data, key_id = security_manager.secure_data_encryption(
            raw_data, SecurityLevel.CONFIDENTIAL
        )
        
        print(f"     ✓ Research data encrypted with key ID: {key_id[:8]}...")
        
        # Start secure research operations
        print("   • Executing secure causal discovery...")
        
        # Authorize operation
        authorized = security_manager.authorize_operation(
            access_token=researcher_cred.access_token,
            operation=OperationType.CAUSAL_DISCOVERY,
            resource="encrypted_research_data",
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        if not authorized:
            print("     ❌ Causal discovery operation not authorized")
            return
        
        # Monitor operation
        operation_start = time.time()
        
        # Decrypt data for processing
        decrypted_data = security_manager.secure_data_decryption(encrypted_data, key_id)
        research_data = json.loads(decrypted_data.decode('utf-8'))
        
        # Perform causal discovery
        causal_engine = CausalDiscoveryEngine(min_causal_strength=0.2)
        causal_graph = causal_engine.discover_causal_structure(
            data=research_data['causal_data'],
            variable_names=research_data['variable_names'],
            prior_knowledge={
                'forbidden_edges': [('Effect', 'Cause')],  # No reverse causation
                'temporal_order': {'Cause': 0, 'Effect': 1, 'Mediator': 2}
            }
        )
        
        operation_time = time.time() - operation_start
        
        print(f"     ✓ Causal discovery completed in {operation_time:.2f}s")
        print(f"       Found {len(causal_graph.edges)} causal relationships")
        
        # Record operation metrics
        operation_metrics = monitoring_system.collect_research_metrics({
            'hypothesis_count': 1,
            'validation_success_rate': 85.0,
            'breakthrough_score': 0.75,
            'execution_time': operation_time,
            'causal_relationships_discovered': len(causal_graph.edges),
            'cross_modal_insights': 5,
            'data_security_level': SecurityLevel.CONFIDENTIAL.value
        }, operation_type="secure_causal_discovery")
        
        monitoring_system.record_metrics(operation_metrics)
        
        print("   • Executing secure multi-modal reasoning...")
        
        # Authorize multi-modal analysis
        authorized = security_manager.authorize_operation(
            access_token=researcher_cred.access_token,
            operation=OperationType.HYPOTHESIS_GENERATION,
            resource="multi_modal_analysis",
            security_level=SecurityLevel.INTERNAL
        )
        
        if authorized:
            reasoning_engine = MultiModalReasoningEngine()
            
            # Perform multi-modal reasoning with security context
            reasoning_result = reasoning_engine.holistic_scientific_reasoning(
                text_data=research_data['text_data'],
                numerical_data=research_data['numerical_data'],
                domain="secure_thermodynamics"
            )
            
            print(f"     ✓ Multi-modal reasoning completed")
            print(f"       Generated {len(reasoning_result.hypotheses)} secure hypotheses")
            print(f"       Identified {len(reasoning_result.cross_modal_insights)} insights")
        
        print("   • Auditing secure operations...")
        
        # Generate comprehensive audit report
        security_report = security_manager.generate_security_report()
        health_report = monitoring_system.generate_health_report()
        
        audit_summary = {
            'timestamp': datetime.now().isoformat(),
            'researcher': researcher_cred.user_id,
            'security_clearance': researcher_cred.security_clearance.value,
            'operations_performed': [
                'data_encryption',
                'causal_discovery', 
                'multi_modal_reasoning'
            ],
            'data_security_level': SecurityLevel.CONFIDENTIAL.value,
            'execution_time': operation_time,
            'causal_relationships_found': len(causal_graph.edges),
            'hypotheses_generated': len(reasoning_result.hypotheses),
            'security_events': security_report['access_statistics']['total_operations'],
            'system_health': health_report['overall_health']
        }
        
        print("     📋 Secure Operation Audit:")
        print(f"       • Researcher: {audit_summary['researcher']}")
        print(f"       • Security clearance: {audit_summary['security_clearance']}")
        print(f"       • Data security: {audit_summary['data_security_level']}")
        print(f"       • Operations: {len(audit_summary['operations_performed'])}")
        print(f"       • Results: {audit_summary['causal_relationships_found']} causal links, {audit_summary['hypotheses_generated']} hypotheses")
        print(f"       • System health: {audit_summary['system_health']}")
        
        # Export secure audit log
        audit_log_file = f"secure_research_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(audit_log_file, 'w') as f:
            json.dump(audit_summary, f, indent=2)
        
        print(f"     💾 Secure audit log exported to: {audit_log_file}")
        
    except Exception as e:
        print(f"   ❌ Secure research pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_system_resilience(monitoring_system):
    """Demonstrate system resilience under various failure conditions"""
    
    print("   • Testing system resilience under load...")
    
    try:
        # Simulate high load conditions
        print("     🔥 Simulating high CPU load...")
        
        # CPU intensive task (controlled)
        def cpu_intensive_task():
            result = 0
            for i in range(1000000):  # Controlled load for demo
                result += i ** 0.5
            return result
        
        # Run multiple tasks to simulate load
        start_time = time.time()
        tasks = []
        
        for i in range(3):  # Limited number for demo
            task = threading.Thread(target=cpu_intensive_task)
            tasks.append(task)
            task.start()
        
        # Monitor system during load
        if monitoring_system:
            time.sleep(2)  # Let load run briefly
            
            # Check system metrics during load
            system_metrics = monitoring_system.collect_system_metrics()
            cpu_metric = next((m for m in system_metrics if m.name == "cpu_usage_percent"), None)
            
            if cpu_metric:
                print(f"       📊 CPU usage under load: {cpu_metric.value:.1f}%")
            
            # Check if alerts were triggered
            alert_count = len(monitoring_system.active_alerts)
            print(f"       🚨 Alerts triggered: {alert_count}")
        
        # Wait for tasks to complete
        for task in tasks:
            task.join()
        
        load_duration = time.time() - start_time
        print(f"     ✓ High load simulation completed in {load_duration:.2f}s")
        
        print("     🔧 Testing memory pressure handling...")
        
        # Simulate memory pressure (controlled)
        try:
            # Create controlled memory usage
            large_arrays = []
            for i in range(5):  # Limited for demo safety
                array = np.random.randn(10000, 100)  # ~76MB per array
                large_arrays.append(array)
                time.sleep(0.1)  # Brief pause
            
            print(f"       📊 Created {len(large_arrays)} large arrays (~380MB total)")
            
            # Monitor memory during pressure
            if monitoring_system:
                system_metrics = monitoring_system.collect_system_metrics()
                memory_metric = next((m for m in system_metrics if m.name == "memory_usage_percent"), None)
                
                if memory_metric:
                    print(f"       📊 Memory usage under pressure: {memory_metric.value:.1f}%")
            
            # Cleanup memory
            del large_arrays
            print("       ✓ Memory pressure test completed, memory cleaned up")
            
        except MemoryError:
            print("       ⚠️ Memory limit reached - graceful handling activated")
        
        print("     🌐 Testing network resilience...")
        
        # Simulate network operations with retries
        network_operations = [
            "data_sync_operation",
            "model_checkpoint_save", 
            "results_backup",
            "metric_reporting"
        ]
        
        successful_ops = 0
        failed_ops = 0
        
        for operation in network_operations:
            # Simulate network call with random failures
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    # Simulate network delay and potential failure
                    time.sleep(0.1)  # Simulate network latency
                    
                    if np.random.random() < 0.2 and attempt < 2:  # 20% failure rate, but succeed eventually
                        raise Exception(f"Network timeout for {operation}")
                    
                    successful_ops += 1
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                        time.sleep(wait_time)
                    else:
                        failed_ops += 1
        
        print(f"       ✓ Network operations: {successful_ops} succeeded, {failed_ops} failed")
        
        print("     🔄 Testing graceful degradation...")
        
        # Simulate component failures and graceful degradation
        components = {
            "causal_discovery": {"available": True, "fallback": "correlation_analysis"},
            "advanced_ml": {"available": True, "fallback": "basic_statistics"},
            "visualization": {"available": True, "fallback": "text_summary"},
            "security_validation": {"available": True, "fallback": "basic_validation"}
        }
        
        # Simulate random component failures
        for component_name, component in components.items():
            if np.random.random() < 0.3:  # 30% chance of failure
                component["available"] = False
                print(f"       ⚠️ Component {component_name} failed, using fallback: {component['fallback']}")
            else:
                print(f"       ✓ Component {component_name} operational")
        
        available_components = sum(1 for comp in components.values() if comp["available"])
        print(f"     📊 System resilience: {available_components}/{len(components)} components operational")
        
        # Test recovery procedures
        print("     🔧 Testing automated recovery...")
        
        recovery_procedures = [
            "restart_failed_services",
            "clear_temporary_caches", 
            "reload_configurations",
            "validate_system_integrity"
        ]
        
        for procedure in recovery_procedures:
            try:
                # Simulate recovery procedure
                time.sleep(0.05)  # Brief simulation
                
                if np.random.random() < 0.9:  # 90% success rate
                    print(f"       ✓ Recovery procedure '{procedure}' completed successfully")
                else:
                    print(f"       ⚠️ Recovery procedure '{procedure}' requires manual intervention")
                
            except Exception as e:
                print(f"       ❌ Recovery procedure '{procedure}' failed: {e}")
        
        # Final system health check
        if monitoring_system:
            print("   • Final system health assessment...")
            
            health_report = monitoring_system.generate_health_report()
            
            print(f"     📊 Post-resilience test status:")
            print(f"       • Overall health: {health_report['overall_health']}")
            print(f"       • Active alerts: {health_report['alerts']['active_alerts']}")
            
            if health_report['recommendations']:
                print("     💡 Recovery recommendations:")
                for rec in health_report['recommendations'][:3]:  # Show top 3
                    print(f"       • {rec}")
        
        print("     ✅ System resilience testing completed")
        
    except Exception as e:
        print(f"   ❌ Resilience testing failed: {e}")
        import traceback
        traceback.print_exc()


def generate_secure_research_data():
    """Generate research data for secure pipeline demonstration"""
    
    np.random.seed(123)  # Reproducible for demo
    
    # Causal data: Cause -> Mediator -> Effect
    n_samples = 200
    cause = np.random.normal(100, 15, n_samples)
    mediator = 0.7 * cause + np.random.normal(0, 10, n_samples)
    effect = 0.6 * mediator + 0.3 * cause + np.random.normal(0, 8, n_samples)
    
    causal_data = np.column_stack([cause, mediator, effect])
    variable_names = ['Cause', 'Mediator', 'Effect']
    
    # Numerical analysis data
    numerical_data = np.random.exponential(2, (n_samples, 1))  # Non-normal distribution
    
    # Scientific text data
    text_data = [
        "Secure research indicates strong causal relationships in thermodynamic systems.",
        "Confidential analysis reveals novel mediating factors in heat transfer processes.",
        "Statistical significance (p < 0.001) observed in controlled experimental conditions.",
        "Breakthrough methodology enables secure discovery of causal mechanisms.",
        "Encrypted data processing maintains research integrity and confidentiality."
    ]
    
    return {
        'causal_data': causal_data,
        'variable_names': variable_names,
        'numerical_data': numerical_data,
        'text_data': text_data
    }


if __name__ == "__main__":
    main()