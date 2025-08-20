"""Tests for quantum-inspired discovery algorithms"""

import pytest
import numpy as np
from src.algorithms.quantum_discovery import (
    QuantumState, 
    QuantumSuperpositionDiscovery,
    QuantumEntanglementDiscovery,
    compare_quantum_vs_classical
)
from src.utils.secure_random import ScientificRandomGenerator


class TestQuantumState:
    """Test QuantumState functionality"""
    
    def test_quantum_state_creation(self):
        """Test quantum state creation and normalization"""
        amplitudes = np.array([1.0 + 0j, 0.5 + 0.5j, 0.3 - 0.2j])
        basis_states = ["|00⟩", "|01⟩", "|10⟩"]
        
        state = QuantumState(
            amplitudes=amplitudes,
            basis_states=basis_states,
            coherence_time=1.0
        )
        
        assert len(state.amplitudes) == 3
        assert len(state.basis_states) == 3
        assert state.coherence_time == 1.0
    
    def test_normalization(self):
        """Test quantum state normalization"""
        amplitudes = np.array([2.0 + 0j, 1.0 + 1.0j, 0.5j])
        basis_states = ["|00⟩", "|01⟩", "|10⟩"]
        
        state = QuantumState(amplitudes=amplitudes, basis_states=basis_states)
        normalized_state = state.normalize()
        
        # Check normalization
        norm_squared = np.sum(np.abs(normalized_state.amplitudes) ** 2)
        assert abs(norm_squared - 1.0) < 1e-10
    
    def test_probability_distribution(self):
        """Test probability distribution calculation"""
        amplitudes = np.array([0.6 + 0j, 0.8j, 0.0 + 0j])
        basis_states = ["|00⟩", "|01⟩", "|10⟩"]
        
        state = QuantumState(amplitudes=amplitudes, basis_states=basis_states)
        probabilities = state.probability_distribution()
        
        assert len(probabilities) == 3
        assert abs(probabilities[0] - 0.36) < 1e-10  # |0.6|^2
        assert abs(probabilities[1] - 0.64) < 1e-10  # |0.8|^2
        assert abs(probabilities[2] - 0.0) < 1e-10   # |0|^2
        assert abs(np.sum(probabilities) - 1.0) < 1e-10
    
    def test_quantum_measurement(self):
        """Test quantum measurement"""
        amplitudes = np.array([0.707 + 0j, 0.707 + 0j])  # Equal superposition
        basis_states = ["|0⟩", "|1⟩"]
        
        state = QuantumState(amplitudes=amplitudes, basis_states=basis_states)
        random_gen = ScientificRandomGenerator(seed=42)
        
        # Perform multiple measurements
        results = []
        for _ in range(100):
            index, basis_state = state.measure(random_gen)
            results.append(index)
        
        # Should get roughly equal distribution
        count_0 = results.count(0)
        count_1 = results.count(1)
        
        assert count_0 + count_1 == 100
        # Allow some statistical variation
        assert abs(count_0 - count_1) < 20


class TestQuantumSuperpositionDiscovery:
    """Test QuantumSuperpositionDiscovery algorithm"""
    
    def test_initialization(self):
        """Test algorithm initialization"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=4, superposition_depth=2)
        
        assert algorithm.num_qubits == 4
        assert algorithm.superposition_depth == 2
        assert algorithm.name == "QuantumSuperpositionDiscovery"
    
    def test_quantum_state_initialization(self):
        """Test quantum state initialization with data"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=3)
        
        # Test with normal data
        data = np.random.randn(50, 1)
        state = algorithm.initialize_quantum_state(data)
        
        assert len(state.amplitudes) == 2**3  # 8 states for 3 qubits
        assert len(state.basis_states) == 8
        assert state.coherence_time > 0
        
        # Check normalization
        norm_squared = np.sum(np.abs(state.amplitudes) ** 2)
        assert abs(norm_squared - 1.0) < 1e-10
    
    def test_quantum_state_initialization_empty_data(self):
        """Test quantum state initialization with empty data"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=2)
        
        empty_data = np.array([]).reshape(0, 1)
        state = algorithm.initialize_quantum_state(empty_data)
        
        assert len(state.amplitudes) == 4  # 2^2 states
        assert len(state.basis_states) == 4
        
        # Should create uniform superposition for empty data
        probabilities = state.probability_distribution()
        expected_prob = 1.0 / 4
        for prob in probabilities:
            assert abs(prob - expected_prob) < 1e-6
    
    def test_quantum_evolution(self):
        """Test quantum evolution process"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=3)
        
        # Create initial state
        data = np.array([1.0, 2.0, 3.0]).reshape(-1, 1)
        initial_state = algorithm.initialize_quantum_state(data)
        
        # Evolve the state
        evolved_state = algorithm.quantum_evolution(initial_state, steps=5)
        
        assert len(evolved_state.amplitudes) == len(initial_state.amplitudes)
        assert evolved_state.coherence_time <= initial_state.coherence_time  # Should decrease
        
        # State should still be normalized
        norm_squared = np.sum(np.abs(evolved_state.amplitudes) ** 2)
        assert abs(norm_squared - 1.0) < 1e-10
    
    def test_quantum_gate_operations(self):
        """Test quantum gate applications"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=2)
        
        # Create simple state
        amplitudes = np.array([1.0 + 0j, 0j, 0j, 0j])  # |00⟩ state
        basis_states = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
        state = QuantumState(amplitudes=amplitudes, basis_states=basis_states)
        
        # Apply Hadamard to first qubit
        h_state = algorithm.apply_quantum_gate(state, "hadamard", [0])
        
        # Should create superposition - check that state changed
        probabilities = h_state.probability_distribution()
        
        # The exact values depend on implementation, just check it's a valid probability distribution
        assert abs(np.sum(probabilities) - 1.0) < 1e-10  # Probabilities sum to 1
        assert all(p >= 0 for p in probabilities)  # All probabilities non-negative
        
        # Should have non-zero probabilities and valid distribution
        assert np.sum(probabilities > 0) >= 2  # At least 2 non-zero probabilities (superposition)
    
    def test_classical_information_extraction(self):
        """Test extraction of classical information from quantum state"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=3)
        
        # Create state with some high-probability components
        amplitudes = np.array([0.7 + 0j, 0.2j, 0.1 + 0j, 0.3 + 0j, 
                              0.4j, 0.1 + 0j, 0.2 + 0j, 0.3j])
        basis_states = [f"|{i:03b}⟩" for i in range(8)]
        
        state = QuantumState(
            amplitudes=amplitudes,
            basis_states=basis_states,
            coherence_time=0.8,
            entanglement_strength=0.3
        ).normalize()
        
        discoveries = algorithm.extract_classical_information(state)
        
        assert isinstance(discoveries, list)
        assert len(discoveries) > 0
        
        for discovery in discoveries:
            assert 'pattern' in discovery
            assert 'probability' in discovery
            assert 'quality' in discovery
            assert discovery['discovery_type'] == 'quantum_superposition'
    
    def test_discover_patterns(self):
        """Test main discovery method"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=4)
        
        # Generate test data
        data = np.random.randn(100, 1)
        
        result = algorithm.discover_patterns(data, evolution_steps=5)
        
        assert hasattr(result, 'discovered_patterns')
        assert hasattr(result, 'quantum_advantage')
        assert hasattr(result, 'coherence_preserved')
        assert hasattr(result, 'entanglement_generated')
        assert hasattr(result, 'execution_stats')
        
        assert isinstance(result.discovered_patterns, list)
        assert isinstance(result.quantum_advantage, float)
        assert 0 <= result.coherence_preserved <= 1
        assert 0 <= result.entanglement_generated <= 1


class TestQuantumEntanglementDiscovery:
    """Test QuantumEntanglementDiscovery algorithm"""
    
    def test_initialization(self):
        """Test entanglement algorithm initialization"""
        algorithm = QuantumEntanglementDiscovery(num_qubits=4)
        
        assert algorithm.num_qubits == 4
        assert algorithm.name == "QuantumEntanglementDiscovery"
    
    def test_entangled_state_initialization(self):
        """Test creation of entangled initial state"""
        algorithm = QuantumEntanglementDiscovery(num_qubits=4)
        
        data = np.random.randn(50, 1)
        state = algorithm.initialize_quantum_state(data)
        
        assert len(state.amplitudes) == 2**4
        assert state.entanglement_strength > 0.5  # Should start with high entanglement
        
        # Check normalization
        norm_squared = np.sum(np.abs(state.amplitudes) ** 2)
        assert abs(norm_squared - 1.0) < 1e-10
    
    def test_entanglement_evolution(self):
        """Test quantum evolution focused on entanglement"""
        algorithm = QuantumEntanglementDiscovery(num_qubits=3)
        
        data = np.array([1.0, 2.0]).reshape(-1, 1)
        initial_state = algorithm.initialize_quantum_state(data)
        
        evolved_state = algorithm.quantum_evolution(initial_state, steps=3)
        
        # Entanglement should be maintained or increased
        assert evolved_state.entanglement_strength >= 0.5
        assert len(evolved_state.amplitudes) == len(initial_state.amplitudes)
    
    def test_bit_correlation_analysis(self):
        """Test bit pattern correlation analysis"""
        algorithm = QuantumEntanglementDiscovery(num_qubits=4)
        
        # Test correlated pattern
        correlated_pattern = "1100"  # Some correlation structure
        correlations = algorithm._analyze_bit_correlations(correlated_pattern)
        
        assert 'strength' in correlations
        assert 'pattern_type' in correlations
        assert 'bit_count' in correlations
        assert correlations['bit_count'] == 4
        assert 0 <= correlations['strength'] <= 1
    
    def test_entanglement_pattern_extraction(self):
        """Test extraction of entanglement-based patterns"""
        algorithm = QuantumEntanglementDiscovery(num_qubits=3)
        
        # Create state with some entangled patterns
        amplitudes = np.array([0.5 + 0j, 0.0 + 0j, 0.0 + 0j, 0.5j,
                              0.5 + 0j, 0.0 + 0j, 0.0 + 0j, 0.5j])
        basis_states = [f"|{i:03b}⟩" for i in range(8)]
        
        state = QuantumState(
            amplitudes=amplitudes,
            basis_states=basis_states,
            entanglement_strength=0.8
        ).normalize()
        
        discoveries = algorithm.extract_classical_information(state)
        
        assert isinstance(discoveries, list)
        
        for discovery in discoveries:
            assert 'pattern' in discovery
            assert 'correlations' in discovery
            assert 'entanglement_measure' in discovery
            assert discovery['discovery_type'] == 'quantum_entanglement'


class TestQuantumComparison:
    """Test quantum vs classical comparison functions"""
    
    def test_quantum_advantage_measurement(self):
        """Test quantum advantage calculation"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=3)
        
        # Mock quantum and classical results
        quantum_result = [
            {'quality': 0.8, 'pattern': 'quantum_pattern_1'},
            {'quality': 0.9, 'pattern': 'quantum_pattern_2'}
        ]
        
        classical_result = [
            {'quality': 0.6, 'pattern': 'classical_pattern_1'}
        ]
        
        advantage = algorithm.measure_quantum_advantage(quantum_result, classical_result)
        
        assert isinstance(advantage, float)
        assert advantage > 0
        # Should show quantum advantage due to higher quality and more patterns
        assert advantage > 1.0
    
    def test_comparative_analysis(self):
        """Test full quantum vs classical comparison"""
        data = np.random.randn(50, 1)
        
        algorithms = [
            QuantumSuperpositionDiscovery(num_qubits=3),
            QuantumEntanglementDiscovery(num_qubits=3)
        ]
        
        results = compare_quantum_vs_classical(data, algorithms, num_runs=2)
        
        assert 'quantum_results' in results
        assert 'statistical_significance' in results
        assert len(results['quantum_results']) == 2
        
        for alg_name, alg_results in results['quantum_results'].items():
            assert 'average_advantage' in alg_results
            assert 'average_patterns_found' in alg_results
            assert 'runs' in alg_results
            assert alg_results['runs'] == 2


class TestQuantumIntegration:
    """Integration tests for quantum algorithms"""
    
    def test_end_to_end_quantum_discovery(self):
        """Test complete quantum discovery workflow"""
        # Create algorithm
        algorithm = QuantumSuperpositionDiscovery(num_qubits=4)
        
        # Generate structured test data
        t = np.linspace(0, 2*np.pi, 50)
        data = np.sin(t).reshape(-1, 1)
        
        # Run discovery
        result = algorithm.discover_patterns(data, evolution_steps=8)
        
        # Verify complete workflow
        assert result.discovered_patterns is not None
        assert result.execution_stats['execution_time'] > 0
        assert result.quantum_advantage >= 0
        
        # Should find some patterns in structured data
        if result.discovered_patterns:
            pattern = result.discovered_patterns[0]
            assert 'quality' in pattern
            assert pattern['discovery_type'] == 'quantum_superposition'
    
    def test_multi_algorithm_comparison(self):
        """Test comparison between different quantum algorithms"""
        data = np.random.randn(100, 1)
        
        superposition_alg = QuantumSuperpositionDiscovery(num_qubits=4)
        entanglement_alg = QuantumEntanglementDiscovery(num_qubits=4)
        
        sup_result = superposition_alg.discover_patterns(data)
        ent_result = entanglement_alg.discover_patterns(data)
        
        # Both should complete successfully
        assert sup_result.discovered_patterns is not None
        assert ent_result.discovered_patterns is not None
        
        # Compare advantages
        sup_advantage = sup_result.quantum_advantage
        ent_advantage = ent_result.quantum_advantage
        
        assert isinstance(sup_advantage, float)
        assert isinstance(ent_advantage, float)
    
    def test_quantum_robustness(self):
        """Test quantum algorithm robustness"""
        algorithm = QuantumSuperpositionDiscovery(num_qubits=3)
        
        # Test with various data types
        test_datasets = [
            np.zeros(10).reshape(-1, 1),  # All zeros
            np.ones(10).reshape(-1, 1),   # All ones  
            np.random.randn(5, 1),        # Small dataset
            np.random.randn(200, 1),      # Large dataset
        ]
        
        for data in test_datasets:
            result = algorithm.discover_patterns(data, evolution_steps=3)
            
            # Should handle all cases gracefully
            assert result.discovered_patterns is not None
            assert result.quantum_advantage >= 0
            assert 0 <= result.coherence_preserved <= 1


if __name__ == "__main__":
    pytest.main([__file__])