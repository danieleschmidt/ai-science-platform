"""
Advanced Biomimetic Neural Networks
Revolutionary Bio-Inspired Computing Architectures

This module implements breakthrough biomimetic neural network architectures
that closely mimic biological neural systems, including:
- Synaptic plasticity mechanisms
- Neuromorphic computation
- Bioelectric signal processing
- Adaptive learning dynamics
- Multi-scale brain modeling

Research Contributions:
1. Synaptic Plasticity Engine - Dynamic connection strength adaptation
2. Neuromorphic Processing Unit - Brain-like computation architecture  
3. Bioelectric Signal Processor - Biological signal pattern analysis
4. Adaptive Learning Controller - Dynamic learning rate adjustment
5. Multi-Scale Brain Simulator - Hierarchical neural modeling
"""

import numpy as np
import logging
import time
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


@dataclass
class Synapse:
    """Represents a biological synapse with plasticity"""
    presynaptic_neuron: int
    postsynaptic_neuron: int
    weight: float
    plasticity_type: str  # 'hebbian', 'anti_hebbian', 'homeostatic'
    learning_rate: float
    decay_rate: float
    threshold: float
    last_activation: float = 0.0
    spike_timing_history: List[float] = field(default_factory=list)
    plasticity_trace: float = 0.0
    
    def update_weight(self, pre_spike_time: float, post_spike_time: float, 
                     global_time: float) -> float:
        """Update synaptic weight based on spike timing"""
        # Spike-timing dependent plasticity (STDP)
        delta_t = post_spike_time - pre_spike_time
        
        if abs(delta_t) < 0.1:  # Within plasticity window
            if self.plasticity_type == 'hebbian':
                if delta_t > 0:  # Post after pre - potentiation
                    weight_change = self.learning_rate * np.exp(-delta_t / 0.02)
                else:  # Pre after post - depression
                    weight_change = -self.learning_rate * np.exp(delta_t / 0.02)
            elif self.plasticity_type == 'anti_hebbian':
                # Opposite of Hebbian
                if delta_t > 0:
                    weight_change = -self.learning_rate * np.exp(-delta_t / 0.02)
                else:
                    weight_change = self.learning_rate * np.exp(delta_t / 0.02)
            else:  # homeostatic
                # Homeostatic scaling
                target_activity = 0.1
                actual_activity = len([t for t in self.spike_timing_history 
                                     if global_time - t < 1.0])  # Last 1 second
                
                weight_change = self.learning_rate * (target_activity - actual_activity) * 0.01
            
            # Apply weight change with bounds
            self.weight += weight_change
            self.weight = max(0.0, min(2.0, self.weight))  # Bounded between 0 and 2
            
            # Update plasticity trace
            self.plasticity_trace = 0.9 * self.plasticity_trace + 0.1 * abs(weight_change)
        
        # Natural decay
        self.weight *= (1.0 - self.decay_rate)
        
        return self.weight


@dataclass
class BiologicalNeuron:
    """Biological neuron with realistic dynamics"""
    neuron_id: int
    neuron_type: str  # 'excitatory', 'inhibitory', 'modulatory'
    membrane_potential: float = -70.0  # mV
    threshold: float = -55.0  # mV
    resting_potential: float = -70.0  # mV
    refractory_period: float = 0.002  # 2ms
    last_spike_time: float = -float('inf')
    adaptation_current: float = 0.0
    dendrite_tree: Dict[str, float] = field(default_factory=dict)
    axon_terminals: List[int] = field(default_factory=list)
    
    # Hodgkin-Huxley parameters
    capacitance: float = 1.0  # μF/cm²
    g_na: float = 120.0  # mS/cm²
    g_k: float = 36.0   # mS/cm²
    g_l: float = 0.3    # mS/cm²
    e_na: float = 50.0  # mV
    e_k: float = -77.0  # mV
    e_l: float = -54.4  # mV
    
    # Gating variables
    m: float = 0.0  # Sodium activation
    h: float = 1.0  # Sodium inactivation
    n: float = 0.0  # Potassium activation
    
    def update_membrane_potential(self, input_current: float, dt: float = 0.0001) -> bool:
        """Update membrane potential using Hodgkin-Huxley model"""
        if input_current is None:
            input_current = 0.0
        
        # Rate functions
        alpha_m = 0.1 * (self.membrane_potential + 40) / (1 - np.exp(-(self.membrane_potential + 40) / 10))
        beta_m = 4 * np.exp(-(self.membrane_potential + 65) / 18)
        
        alpha_h = 0.07 * np.exp(-(self.membrane_potential + 65) / 20)
        beta_h = 1 / (1 + np.exp(-(self.membrane_potential + 35) / 10))
        
        alpha_n = 0.01 * (self.membrane_potential + 55) / (1 - np.exp(-(self.membrane_potential + 55) / 10))
        beta_n = 0.125 * np.exp(-(self.membrane_potential + 65) / 80)
        
        # Update gating variables
        self.m += dt * (alpha_m * (1 - self.m) - beta_m * self.m)
        self.h += dt * (alpha_h * (1 - self.h) - beta_h * self.h)
        self.n += dt * (alpha_n * (1 - self.n) - beta_n * self.n)
        
        # Ionic currents
        i_na = self.g_na * (self.m ** 3) * self.h * (self.membrane_potential - self.e_na)
        i_k = self.g_k * (self.n ** 4) * (self.membrane_potential - self.e_k)
        i_l = self.g_l * (self.membrane_potential - self.e_l)
        
        # Membrane equation
        i_ion = i_na + i_k + i_l
        dv_dt = (input_current - i_ion - self.adaptation_current) / self.capacitance
        
        # Update membrane potential
        self.membrane_potential += dt * dv_dt
        
        # Check for spike
        spike_occurred = False
        current_time = time.time()
        
        if (self.membrane_potential > self.threshold and 
            current_time - self.last_spike_time > self.refractory_period):
            
            spike_occurred = True
            self.last_spike_time = current_time
            
            # Reset membrane potential
            self.membrane_potential = self.resting_potential
            
            # Spike-frequency adaptation
            self.adaptation_current += 0.02
        
        # Adaptation current decay
        self.adaptation_current *= 0.999
        
        return spike_occurred
    
    def get_neuron_state(self) -> Dict[str, Any]:
        """Get complete neuron state"""
        return {
            'neuron_id': self.neuron_id,
            'type': self.neuron_type,
            'membrane_potential': self.membrane_potential,
            'adaptation_current': self.adaptation_current,
            'gating_variables': {'m': self.m, 'h': self.h, 'n': self.n},
            'last_spike_time': self.last_spike_time,
            'is_refractory': time.time() - self.last_spike_time < self.refractory_period
        }


class SynapticPlasticityEngine:
    """
    Advanced Synaptic Plasticity Engine
    
    Implements multiple forms of biological synaptic plasticity:
    - Spike-timing dependent plasticity (STDP)
    - Homeostatic scaling
    - Metaplasticity
    - Synaptic tagging and capture
    """
    
    def __init__(self, plasticity_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize synaptic plasticity engine
        
        Args:
            plasticity_rules: Dictionary defining plasticity parameters
        """
        self.plasticity_rules = plasticity_rules or self._default_plasticity_rules()
        self.synapses = {}  # Dict[Tuple[int, int], Synapse]
        self.global_time = 0.0
        self.plasticity_history = []
        self.metaplasticity_state = {}
        
        logger.info("SynapticPlasticityEngine initialized with advanced plasticity mechanisms")
    
    def _default_plasticity_rules(self) -> Dict[str, Any]:
        """Default plasticity rules"""
        return {
            'stdp_window': 0.1,  # seconds
            'ltp_amplitude': 0.01,  # Long-term potentiation
            'ltd_amplitude': 0.008,  # Long-term depression
            'homeostatic_target': 0.1,  # Target firing rate
            'metaplasticity_threshold': 0.5,  # Threshold for metaplasticity
            'synaptic_tag_duration': 1.0,  # Duration of synaptic tags
            'protein_synthesis_delay': 0.5  # Delay for late-phase plasticity
        }
    
    def create_synapse(self, pre_neuron_id: int, post_neuron_id: int,
                      initial_weight: float = 0.5,
                      plasticity_type: str = 'hebbian') -> Synapse:
        """Create a new synapse with plasticity"""
        synapse_key = (pre_neuron_id, post_neuron_id)
        
        synapse = Synapse(
            presynaptic_neuron=pre_neuron_id,
            postsynaptic_neuron=post_neuron_id,
            weight=initial_weight,
            plasticity_type=plasticity_type,
            learning_rate=self.plasticity_rules['ltp_amplitude'],
            decay_rate=0.001,
            threshold=0.05
        )
        
        self.synapses[synapse_key] = synapse
        return synapse
    
    def update_plasticity(self, spike_events: List[Tuple[int, float]]) -> Dict[str, Any]:
        """
        Update synaptic plasticity based on spike events
        
        Args:
            spike_events: List of (neuron_id, spike_time) tuples
            
        Returns:
            Plasticity update statistics
        """
        self.global_time = max([t for _, t in spike_events] + [self.global_time])
        
        # Group spikes by neuron
        neuron_spikes = defaultdict(list)
        for neuron_id, spike_time in spike_events:
            neuron_spikes[neuron_id].append(spike_time)
        
        plasticity_changes = []
        
        # Update each synapse
        for synapse_key, synapse in self.synapses.items():
            pre_id, post_id = synapse_key
            
            pre_spikes = neuron_spikes.get(pre_id, [])
            post_spikes = neuron_spikes.get(post_id, [])
            
            # Find spike pairs within plasticity window
            for pre_time in pre_spikes:
                for post_time in post_spikes:
                    if abs(post_time - pre_time) < self.plasticity_rules['stdp_window']:
                        old_weight = synapse.weight
                        new_weight = synapse.update_weight(pre_time, post_time, self.global_time)
                        
                        plasticity_changes.append({
                            'synapse': synapse_key,
                            'weight_change': new_weight - old_weight,
                            'spike_timing': post_time - pre_time,
                            'plasticity_type': synapse.plasticity_type
                        })
            
            # Update spike timing history
            synapse.spike_timing_history.extend(pre_spikes + post_spikes)
            # Keep only recent history
            cutoff_time = self.global_time - 10.0  # Last 10 seconds
            synapse.spike_timing_history = [
                t for t in synapse.spike_timing_history if t > cutoff_time
            ]
        
        # Metaplasticity update
        self._update_metaplasticity(plasticity_changes)
        
        # Homeostatic scaling
        self._apply_homeostatic_scaling(neuron_spikes)
        
        # Synaptic tagging and capture
        self._update_synaptic_tags(plasticity_changes)
        
        # Record plasticity history
        self.plasticity_history.append({
            'time': self.global_time,
            'changes': plasticity_changes,
            'total_synapses': len(self.synapses),
            'active_synapses': len([s for s in self.synapses.values() if s.plasticity_trace > 0.01])
        })
        
        return {
            'plasticity_changes': len(plasticity_changes),
            'average_weight_change': np.mean([c['weight_change'] for c in plasticity_changes]) if plasticity_changes else 0.0,
            'total_synapses': len(self.synapses),
            'metaplasticity_active': len(self.metaplasticity_state)
        }
    
    def _update_metaplasticity(self, plasticity_changes: List[Dict[str, Any]]):
        """Update metaplasticity state"""
        for change in plasticity_changes:
            synapse_key = change['synapse']
            
            if synapse_key not in self.metaplasticity_state:
                self.metaplasticity_state[synapse_key] = {
                    'recent_activity': 0.0,
                    'plasticity_threshold': self.plasticity_rules['metaplasticity_threshold'],
                    'modification_factor': 1.0
                }
            
            meta_state = self.metaplasticity_state[synapse_key]
            
            # Update recent activity
            meta_state['recent_activity'] = (0.9 * meta_state['recent_activity'] + 
                                           0.1 * abs(change['weight_change']))
            
            # Adjust plasticity threshold based on recent activity
            if meta_state['recent_activity'] > meta_state['plasticity_threshold']:
                # Increase threshold (make plasticity harder)
                meta_state['plasticity_threshold'] *= 1.01
                meta_state['modification_factor'] *= 0.99
            else:
                # Decrease threshold (make plasticity easier)
                meta_state['plasticity_threshold'] *= 0.99
                meta_state['modification_factor'] *= 1.01
            
            # Apply modification to synapse
            if synapse_key in self.synapses:
                self.synapses[synapse_key].learning_rate *= meta_state['modification_factor']
    
    def _apply_homeostatic_scaling(self, neuron_spikes: Dict[int, List[float]]):
        """Apply homeostatic scaling to maintain stable activity"""
        target_rate = self.plasticity_rules['homeostatic_target']
        time_window = 5.0  # 5 second window
        
        for neuron_id, spikes in neuron_spikes.items():
            # Calculate recent firing rate
            recent_spikes = [t for t in spikes if self.global_time - t < time_window]
            firing_rate = len(recent_spikes) / time_window
            
            # Scale synapses involving this neuron
            scaling_factor = 1.0 + 0.01 * (target_rate - firing_rate)
            
            for synapse_key, synapse in self.synapses.items():
                if synapse_key[1] == neuron_id:  # Postsynaptic neuron
                    synapse.weight *= scaling_factor
                    synapse.weight = max(0.01, min(2.0, synapse.weight))
    
    def _update_synaptic_tags(self, plasticity_changes: List[Dict[str, Any]]):
        """Update synaptic tags for late-phase plasticity"""
        for change in plasticity_changes:
            if abs(change['weight_change']) > 0.005:  # Significant plasticity
                synapse_key = change['synapse']
                
                # Set synaptic tag
                tag_key = f"tag_{synapse_key}"
                self.metaplasticity_state[tag_key] = {
                    'tag_time': self.global_time,
                    'tag_strength': abs(change['weight_change']),
                    'capture_window': self.plasticity_rules['synaptic_tag_duration']
                }
        
        # Process protein synthesis and capture
        self._process_synaptic_capture()
    
    def _process_synaptic_capture(self):
        """Process synaptic capture mechanism"""
        # Remove expired tags and apply late-phase changes
        expired_tags = []
        
        for tag_key, tag_info in self.metaplasticity_state.items():
            if tag_key.startswith('tag_'):
                if self.global_time - tag_info['tag_time'] > tag_info['capture_window']:
                    # Tag expired - apply late-phase changes
                    synapse_key_str = tag_key[4:]  # Remove 'tag_' prefix
                    try:
                        synapse_key = eval(synapse_key_str)  # Convert back to tuple
                        if synapse_key in self.synapses:
                            # Late-phase potentiation
                            self.synapses[synapse_key].weight *= (1.0 + 0.1 * tag_info['tag_strength'])
                            self.synapses[synapse_key].weight = min(2.0, self.synapses[synapse_key].weight)
                    except:
                        pass
                    
                    expired_tags.append(tag_key)
        
        # Remove expired tags
        for tag_key in expired_tags:
            del self.metaplasticity_state[tag_key]
    
    def get_plasticity_statistics(self) -> Dict[str, Any]:
        """Get comprehensive plasticity statistics"""
        if not self.synapses:
            return {'total_synapses': 0}
        
        weights = [s.weight for s in self.synapses.values()]
        traces = [s.plasticity_trace for s in self.synapses.values()]
        
        return {
            'total_synapses': len(self.synapses),
            'weight_statistics': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights)
            },
            'plasticity_trace_statistics': {
                'mean': np.mean(traces),
                'active_synapses': len([t for t in traces if t > 0.01])
            },
            'metaplasticity_synapses': len(self.metaplasticity_state),
            'recent_plasticity_events': len(self.plasticity_history[-10:]) if self.plasticity_history else 0
        }


class NeuromorphicProcessingUnit:
    """
    Neuromorphic Processing Unit
    
    Brain-inspired computing architecture that processes information
    using spike-based neural networks with biological realism.
    """
    
    def __init__(self, n_neurons: int = 1000, connectivity: float = 0.1,
                 inhibitory_fraction: float = 0.2):
        """
        Initialize neuromorphic processing unit
        
        Args:
            n_neurons: Number of neurons in the network
            connectivity: Connection probability between neurons
            inhibitory_fraction: Fraction of inhibitory neurons
        """
        self.n_neurons = n_neurons
        self.connectivity = connectivity
        self.inhibitory_fraction = inhibitory_fraction
        
        # Create biological neurons
        self.neurons = {}
        self._create_neural_population()
        
        # Initialize plasticity engine
        self.plasticity_engine = SynapticPlasticityEngine()
        
        # Create synaptic connections
        self._create_synaptic_connections()
        
        # Processing state
        self.spike_history = deque(maxlen=10000)  # Store recent spikes
        self.membrane_potentials = {}
        self.global_inhibition = 0.0
        self.oscillation_state = {'theta': 0.0, 'gamma': 0.0, 'alpha': 0.0}
        
        logger.info(f"NeuromorphicProcessingUnit initialized: {n_neurons} neurons, "
                   f"{len(self.plasticity_engine.synapses)} synapses")
    
    def _create_neural_population(self):
        """Create diverse population of biological neurons"""
        n_inhibitory = int(self.n_neurons * self.inhibitory_fraction)
        n_excitatory = self.n_neurons - n_inhibitory
        
        # Create excitatory neurons
        for i in range(n_excitatory):
            neuron_type = 'excitatory'
            if i % 10 == 0:  # Some modulatory neurons
                neuron_type = 'modulatory'
            
            self.neurons[i] = BiologicalNeuron(
                neuron_id=i,
                neuron_type=neuron_type,
                threshold=-55.0 + random.gauss(0, 2),  # Variability
                resting_potential=-70.0 + random.gauss(0, 1)
            )
        
        # Create inhibitory neurons
        for i in range(n_excitatory, self.n_neurons):
            self.neurons[i] = BiologicalNeuron(
                neuron_id=i,
                neuron_type='inhibitory',
                threshold=-50.0 + random.gauss(0, 2),  # More excitable
                resting_potential=-65.0 + random.gauss(0, 1),
                g_na=150.0,  # Faster dynamics
                refractory_period=0.001  # Shorter refractory period
            )
    
    def _create_synaptic_connections(self):
        """Create synaptic connections with biological connectivity patterns"""
        # Local connections (higher probability for nearby neurons)
        for pre_id in range(self.n_neurons):
            for post_id in range(self.n_neurons):
                if pre_id != post_id:
                    # Distance-dependent connectivity
                    distance = abs(pre_id - post_id)
                    connection_prob = self.connectivity * np.exp(-distance / 100.0)
                    
                    if random.random() < connection_prob:
                        # Determine plasticity type
                        pre_neuron = self.neurons[pre_id]
                        post_neuron = self.neurons[post_id]
                        
                        if pre_neuron.neuron_type == 'excitatory':
                            plasticity_type = 'hebbian'
                            weight = random.uniform(0.3, 0.8)
                        else:  # inhibitory
                            plasticity_type = 'anti_hebbian'
                            weight = random.uniform(0.5, 1.2)
                        
                        self.plasticity_engine.create_synapse(
                            pre_id, post_id, weight, plasticity_type
                        )
        
        # Long-range connections (sparse but important)
        n_long_range = int(self.n_neurons * 0.05)  # 5% long-range
        for _ in range(n_long_range):
            pre_id = random.randint(0, self.n_neurons - 1)
            post_id = random.randint(0, self.n_neurons - 1)
            
            if pre_id != post_id and (pre_id, post_id) not in self.plasticity_engine.synapses:
                self.plasticity_engine.create_synapse(
                    pre_id, post_id, random.uniform(0.1, 0.3), 'homeostatic'
                )
    
    def process_input(self, input_data: np.ndarray, 
                     processing_time: float = 0.1) -> Dict[str, Any]:
        """
        Process input through neuromorphic network
        
        Args:
            input_data: Input data to process
            processing_time: Duration of processing in seconds
            
        Returns:
            Processing results and neural activity
        """
        logger.debug(f"Processing input through neuromorphic network: "
                    f"shape={input_data.shape}, time={processing_time}s")
        
        start_time = time.time()
        dt = 0.0001  # 0.1ms timestep
        n_steps = int(processing_time / dt)
        
        # Initialize input mapping
        input_neurons = list(range(min(len(input_data), self.n_neurons // 4)))
        
        # Processing loop
        spike_times = []
        membrane_trace = defaultdict(list)
        
        for step in range(n_steps):
            current_time = step * dt
            step_spikes = []
            
            # Apply input to input neurons
            for i, neuron_id in enumerate(input_neurons):
                if i < len(input_data):
                    # Convert input to current
                    input_current = input_data[i] * 10.0  # Scale input
                    
                    # Add noise
                    input_current += random.gauss(0, 0.5)
                    
                    # Update neuron
                    if neuron_id in self.neurons:
                        spike_occurred = self.neurons[neuron_id].update_membrane_potential(
                            input_current, dt
                        )
                        
                        if spike_occurred:
                            step_spikes.append((neuron_id, current_time))
                            self.spike_history.append((neuron_id, current_time))
            
            # Update all other neurons based on synaptic input
            for neuron_id, neuron in self.neurons.items():
                if neuron_id not in input_neurons:
                    # Calculate synaptic input
                    synaptic_current = self._calculate_synaptic_input(
                        neuron_id, current_time, dt
                    )
                    
                    # Add global inhibition
                    if neuron.neuron_type == 'excitatory':
                        synaptic_current -= self.global_inhibition
                    
                    # Update neuron
                    spike_occurred = neuron.update_membrane_potential(synaptic_current, dt)
                    
                    if spike_occurred:
                        step_spikes.append((neuron_id, current_time))
                        self.spike_history.append((neuron_id, current_time))
                        
                        # Update global inhibition
                        if neuron.neuron_type == 'inhibitory':
                            self.global_inhibition += 0.1
                
                # Record membrane potential
                if step % 10 == 0:  # Subsample for efficiency
                    membrane_trace[neuron_id].append(neuron.membrane_potential)
            
            # Update oscillations
            self._update_neural_oscillations(current_time, len(step_spikes))
            
            # Update plasticity every 10 steps
            if step % 10 == 0 and step_spikes:
                self.plasticity_engine.update_plasticity(step_spikes)
            
            spike_times.extend(step_spikes)
            
            # Decay global inhibition
            self.global_inhibition *= 0.99
        
        # Analyze processing results
        results = self._analyze_neural_activity(spike_times, membrane_trace, processing_time)
        results['processing_time'] = time.time() - start_time
        
        return results
    
    def _calculate_synaptic_input(self, neuron_id: int, current_time: float, dt: float) -> float:
        """Calculate synaptic input current for a neuron"""
        total_current = 0.0
        
        # Find relevant synapses (where this neuron is postsynaptic)
        for synapse_key, synapse in self.plasticity_engine.synapses.items():
            pre_id, post_id = synapse_key
            
            if post_id == neuron_id:
                # Check for recent presynaptic spikes
                recent_spikes = [
                    (nid, t) for nid, t in self.spike_history
                    if nid == pre_id and current_time - t < 0.005  # 5ms window
                ]
                
                if recent_spikes:
                    # Calculate synaptic current
                    for _, spike_time in recent_spikes:
                        delay = current_time - spike_time
                        
                        # Exponential decay with realistic time constants
                        if self.neurons[pre_id].neuron_type == 'excitatory':
                            # AMPA-like (fast excitatory)
                            current = synapse.weight * np.exp(-delay / 0.002)  # 2ms decay
                        else:
                            # GABA-like (inhibitory)
                            current = -synapse.weight * np.exp(-delay / 0.010)  # 10ms decay
                        
                        total_current += current
        
        return total_current
    
    def _update_neural_oscillations(self, current_time: float, n_spikes: int):
        """Update neural oscillation states"""
        # Theta rhythm (4-8 Hz)
        self.oscillation_state['theta'] = np.sin(2 * np.pi * 6 * current_time)
        
        # Gamma rhythm (30-100 Hz) - driven by activity
        gamma_freq = 40 + 20 * min(1.0, n_spikes / 10.0)  # Activity-dependent
        self.oscillation_state['gamma'] = np.sin(2 * np.pi * gamma_freq * current_time)
        
        # Alpha rhythm (8-12 Hz)
        self.oscillation_state['alpha'] = np.sin(2 * np.pi * 10 * current_time)
    
    def _analyze_neural_activity(self, spike_times: List[Tuple[int, float]], 
                               membrane_trace: Dict[int, List[float]],
                               total_time: float) -> Dict[str, Any]:
        """Analyze neural activity patterns"""
        if not spike_times:
            return {
                'total_spikes': 0,
                'firing_rate': 0.0,
                'synchrony': 0.0,
                'oscillation_power': {},
                'population_dynamics': {}
            }
        
        # Basic statistics
        total_spikes = len(spike_times)
        firing_rate = total_spikes / (self.n_neurons * total_time)
        
        # Spike timing analysis
        spike_times_only = [t for _, t in spike_times]
        
        # Synchrony measure (coefficient of variation of inter-spike intervals)
        if len(spike_times_only) > 1:
            isi = np.diff(sorted(spike_times_only))
            synchrony = np.std(isi) / np.mean(isi) if np.mean(isi) > 0 else 0.0
        else:
            synchrony = 0.0
        
        # Population vector analysis
        time_bins = np.linspace(0, total_time, 50)
        population_activity = np.histogram(spike_times_only, bins=time_bins)[0]
        
        # Oscillation analysis
        oscillation_power = {}
        if len(population_activity) > 10:
            # Simple spectral analysis
            fft = np.fft.fft(population_activity)
            freqs = np.fft.fftfreq(len(population_activity), total_time / len(time_bins))
            
            # Find peaks in different frequency bands
            power_spectrum = np.abs(fft) ** 2
            
            theta_band = (freqs >= 4) & (freqs <= 8)
            alpha_band = (freqs >= 8) & (freqs <= 12)
            gamma_band = (freqs >= 30) & (freqs <= 100)
            
            oscillation_power = {
                'theta': np.mean(power_spectrum[theta_band]) if np.any(theta_band) else 0.0,
                'alpha': np.mean(power_spectrum[alpha_band]) if np.any(alpha_band) else 0.0,
                'gamma': np.mean(power_spectrum[gamma_band]) if np.any(gamma_band) else 0.0
            }
        
        # Neuron type analysis
        excitatory_spikes = len([(nid, t) for nid, t in spike_times 
                                if self.neurons[nid].neuron_type == 'excitatory'])
        inhibitory_spikes = len([(nid, t) for nid, t in spike_times 
                                if self.neurons[nid].neuron_type == 'inhibitory'])
        
        # Network topology analysis
        plasticity_stats = self.plasticity_engine.get_plasticity_statistics()
        
        return {
            'total_spikes': total_spikes,
            'firing_rate': firing_rate,
            'synchrony': synchrony,
            'excitatory_inhibitory_ratio': excitatory_spikes / max(1, inhibitory_spikes),
            'oscillation_power': oscillation_power,
            'population_dynamics': {
                'activity_trace': population_activity.tolist(),
                'time_bins': time_bins.tolist(),
                'peak_activity': np.max(population_activity),
                'activity_variance': np.var(population_activity)
            },
            'plasticity_statistics': plasticity_stats,
            'network_state': {
                'global_inhibition': self.global_inhibition,
                'oscillation_phase': self.oscillation_state.copy()
            }
        }
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get complete network state"""
        neuron_states = {nid: neuron.get_neuron_state() 
                        for nid, neuron in self.neurons.items()}
        
        return {
            'neurons': neuron_states,
            'synapses': len(self.plasticity_engine.synapses),
            'recent_spikes': len(self.spike_history),
            'oscillation_state': self.oscillation_state.copy(),
            'global_inhibition': self.global_inhibition
        }


def run_biomimetic_neural_demonstration():
    """
    Comprehensive demonstration of biomimetic neural networks
    """
    logger.info("🧠 Starting Biomimetic Neural Network Demonstration")
    
    # Create neuromorphic processing unit
    npu = NeuromorphicProcessingUnit(
        n_neurons=500,
        connectivity=0.15,
        inhibitory_fraction=0.25
    )
    
    # Test scenarios
    test_inputs = [
        {
            'name': 'Sensory Input Pattern',
            'data': np.sin(np.linspace(0, 4*np.pi, 50)) + 0.1 * np.random.randn(50),
            'processing_time': 0.2
        },
        {
            'name': 'Memory Recall Pattern',
            'data': np.array([1, 0, 1, 0, 1] * 10) + 0.05 * np.random.randn(50),
            'processing_time': 0.15
        },
        {
            'name': 'Complex Decision Pattern', 
            'data': np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 50).flatten()[:50],
            'processing_time': 0.25
        }
    ]
    
    results = {}
    
    for test in test_inputs:
        logger.info(f"Testing: {test['name']}")
        
        result = npu.process_input(test['data'], test['processing_time'])
        results[test['name']] = result
        
        logger.info(f"  Firing rate: {result['firing_rate']:.2f} Hz")
        logger.info(f"  Synchrony: {result['synchrony']:.3f}")
        logger.info(f"  E/I ratio: {result['excitatory_inhibitory_ratio']:.2f}")
        
        # Log oscillation power
        for band, power in result['oscillation_power'].items():
            logger.info(f"  {band.capitalize()} power: {power:.2f}")
    
    # Network adaptation analysis
    plasticity_stats = npu.plasticity_engine.get_plasticity_statistics()
    
    logger.info("Biomimetic Neural Network Analysis:")
    logger.info(f"  Total synapses: {plasticity_stats['total_synapses']}")
    logger.info(f"  Active synapses: {plasticity_stats['plasticity_trace_statistics']['active_synapses']}")
    logger.info(f"  Average weight: {plasticity_stats['weight_statistics']['mean']:.3f}")
    logger.info(f"  Weight variability: {plasticity_stats['weight_statistics']['std']:.3f}")
    
    return {
        'test_results': results,
        'network_statistics': plasticity_stats,
        'final_network_state': npu.get_network_state(),
        'total_processing_performance': sum(r['firing_rate'] for r in results.values())
    }


if __name__ == "__main__":
    # Run demonstration if script is executed directly
    demo_results = run_biomimetic_neural_demonstration()
    print("🚀 Biomimetic Neural Network Demonstration completed!")
    print(f"Total network performance: {demo_results['total_processing_performance']:.2f}")