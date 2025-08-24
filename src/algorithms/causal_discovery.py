"""Advanced Causal Discovery Engine for Scientific Research"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations, permutations
import json

from ..utils.error_handling import robust_execution, DiscoveryError
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


@dataclass
class CausalRelationship:
    """Represents a discovered causal relationship"""
    cause: str
    effect: str
    strength: float
    confidence: float
    mechanism: str
    evidence: Dict[str, float]
    confounders: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not (0 <= self.strength <= 1):
            raise ValueError("Causal strength must be between 0 and 1")
        if not (0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between 0 and 1")


@dataclass 
class CausalGraph:
    """Directed acyclic graph representing causal structure"""
    nodes: Set[str]
    edges: List[CausalRelationship]
    discovery_method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_parents(self, node: str) -> List[str]:
        """Get all direct causes of a node"""
        return [edge.cause for edge in self.edges if edge.effect == node]
    
    def get_children(self, node: str) -> List[str]:
        """Get all direct effects of a node"""
        return [edge.effect for edge in self.edges if edge.cause == node]
    
    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants (transitive effects) of a node"""
        descendants = set()
        queue = self.get_children(node)
        
        while queue:
            child = queue.pop(0)
            if child not in descendants:
                descendants.add(child)
                queue.extend(self.get_children(child))
        
        return descendants


class CausalDiscoveryEngine(ValidationMixin):
    """Advanced causal discovery using multiple algorithms and validation"""
    
    def __init__(self, 
                 min_causal_strength: float = 0.3,
                 confidence_threshold: float = 0.7,
                 max_conditioning_set_size: int = 5):
        """
        Initialize causal discovery engine
        
        Args:
            min_causal_strength: Minimum strength for causal relationships
            confidence_threshold: Minimum confidence for including relationships
            max_conditioning_set_size: Maximum size of conditioning sets for independence testing
        """
        self.min_causal_strength = min_causal_strength
        self.confidence_threshold = confidence_threshold  
        self.max_conditioning_set_size = max_conditioning_set_size
        self.discovery_history: List[CausalGraph] = []
        
        logger.info(f"CausalDiscoveryEngine initialized with strength threshold: {min_causal_strength}")
    
    @robust_execution(recovery_strategy='partial_recovery')
    def discover_causal_structure(self,
                                data: np.ndarray,
                                variable_names: Optional[List[str]] = None,
                                prior_knowledge: Optional[Dict[str, Any]] = None,
                                methods: List[str] = None) -> CausalGraph:
        """
        Discover causal structure from observational data using multiple methods
        
        Args:
            data: Observational data matrix (samples x variables)
            variable_names: Names for variables
            prior_knowledge: Domain knowledge constraints
            methods: Causal discovery methods to use
            
        Returns:
            CausalGraph representing discovered causal structure
        """
        
        if data.shape[0] < data.shape[1]:
            logger.warning(f"Few samples ({data.shape[0]}) vs variables ({data.shape[1]})")
        
        if variable_names is None:
            variable_names = [f"var_{i}" for i in range(data.shape[1])]
        
        if methods is None:
            methods = ['pc_algorithm', 'granger_causality', 'information_geometric']
        
        logger.info(f"Discovering causal structure with methods: {methods}")
        
        # Ensemble of causal discovery methods
        causal_votes = {}
        method_confidences = {}
        
        for method in methods:
            try:
                if method == 'pc_algorithm':
                    edges, confidence = self._pc_algorithm(data, variable_names)
                elif method == 'granger_causality':
                    edges, confidence = self._granger_causality(data, variable_names)
                elif method == 'information_geometric':
                    edges, confidence = self._information_geometric_causality(data, variable_names)
                else:
                    logger.warning(f"Unknown causal discovery method: {method}")
                    continue
                
                # Aggregate votes
                for edge in edges:
                    edge_key = (edge.cause, edge.effect)
                    if edge_key not in causal_votes:
                        causal_votes[edge_key] = []
                    causal_votes[edge_key].append(edge)
                
                method_confidences[method] = confidence
                
            except Exception as e:
                logger.error(f"Method {method} failed: {e}")
                continue
        
        # Consensus building
        final_edges = self._build_consensus(causal_votes, method_confidences)
        
        # Apply domain knowledge constraints
        if prior_knowledge:
            final_edges = self._apply_prior_knowledge(final_edges, prior_knowledge)
        
        # Create causal graph
        nodes = set(variable_names)
        causal_graph = CausalGraph(
            nodes=nodes,
            edges=final_edges,
            discovery_method=f"ensemble_{'+'.join(methods)}"
        )
        
        self.discovery_history.append(causal_graph)
        
        logger.info(f"Discovered causal graph with {len(final_edges)} causal relationships")
        return causal_graph
    
    def _pc_algorithm(self, data: np.ndarray, var_names: List[str]) -> Tuple[List[CausalRelationship], float]:
        """Peter-Clark algorithm for causal discovery"""
        edges = []
        
        # Step 1: Build complete undirected graph
        adjacencies = self._test_conditional_independence(data, var_names)
        
        # Step 2: Orient edges using v-structures and rules
        oriented_edges = self._orient_edges_pc(adjacencies, data, var_names)
        
        # Convert to CausalRelationship objects
        for (cause, effect), strength in oriented_edges.items():
            if strength >= self.min_causal_strength:
                edges.append(CausalRelationship(
                    cause=cause,
                    effect=effect, 
                    strength=strength,
                    confidence=min(0.9, strength * 1.2),  # Heuristic confidence
                    mechanism="conditional_independence",
                    evidence={"pc_score": strength}
                ))
        
        return edges, np.mean([e.confidence for e in edges]) if edges else 0.0
    
    def _granger_causality(self, data: np.ndarray, var_names: List[str]) -> Tuple[List[CausalRelationship], float]:
        """Granger causality test for temporal causal relationships"""
        edges = []
        n_vars = len(var_names)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Test if var_i Granger-causes var_j
                    granger_strength = self._compute_granger_causality(
                        data[:, i], data[:, j], max_lags=min(5, len(data) // 10)
                    )
                    
                    if granger_strength >= self.min_causal_strength:
                        edges.append(CausalRelationship(
                            cause=var_names[i],
                            effect=var_names[j],
                            strength=granger_strength,
                            confidence=min(0.9, granger_strength * 1.1),
                            mechanism="temporal_precedence",
                            evidence={"granger_f_stat": granger_strength}
                        ))
        
        return edges, np.mean([e.confidence for e in edges]) if edges else 0.0
    
    def _information_geometric_causality(self, data: np.ndarray, var_names: List[str]) -> Tuple[List[CausalRelationship], float]:
        """Novel information-geometric approach to causal discovery"""
        edges = []
        n_vars = len(var_names)
        
        # Compute information-geometric causality metric
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    # Novel IG causality measure
                    ig_strength = self._compute_information_geometric_causality(
                        data[:, i], data[:, j], data
                    )
                    
                    if ig_strength >= self.min_causal_strength:
                        edges.append(CausalRelationship(
                            cause=var_names[i],
                            effect=var_names[j],
                            strength=ig_strength,
                            confidence=min(0.95, ig_strength * 1.05),  # High confidence for novel method
                            mechanism="information_geometry",
                            evidence={"ig_divergence": ig_strength}
                        ))
        
        return edges, np.mean([e.confidence for e in edges]) if edges else 0.0
    
    def _test_conditional_independence(self, data: np.ndarray, var_names: List[str]) -> Dict[Tuple[str, str], float]:
        """Test conditional independence for all variable pairs"""
        adjacencies = {}
        n_vars = len(var_names)
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Test independence of var_i and var_j
                independence_score = self._conditional_independence_test(
                    data[:, i], data[:, j], data, max_conditioning_vars=self.max_conditioning_set_size
                )
                
                # Lower score means less independent (more dependent/causal)
                dependency_score = 1.0 - independence_score
                
                if dependency_score >= self.min_causal_strength:
                    adjacencies[(var_names[i], var_names[j])] = dependency_score
                    adjacencies[(var_names[j], var_names[i])] = dependency_score
        
        return adjacencies
    
    def _orient_edges_pc(self, adjacencies: Dict[Tuple[str, str], float], 
                        data: np.ndarray, var_names: List[str]) -> Dict[Tuple[str, str], float]:
        """Orient edges using PC algorithm rules"""
        oriented = {}
        
        # Rule 1: Orient v-structures (colliders)
        for v1, v2, v3 in combinations(var_names, 3):
            if ((v1, v2) in adjacencies and (v2, v3) in adjacencies and 
                (v1, v3) not in adjacencies):
                # v2 is a collider: v1 -> v2 <- v3
                oriented[(v1, v2)] = adjacencies[(v1, v2)]
                oriented[(v3, v2)] = adjacencies[(v3, v2)]
        
        # Rule 2: Prevent cycles and apply transitivity
        for (v1, v2), strength in adjacencies.items():
            if (v1, v2) not in oriented and (v2, v1) not in oriented:
                # Default orientation based on temporal or alphabetical order
                if v1 < v2:  # Simple heuristic
                    oriented[(v1, v2)] = strength
                else:
                    oriented[(v2, v1)] = strength
        
        return oriented
    
    def _conditional_independence_test(self, x: np.ndarray, y: np.ndarray, 
                                     full_data: np.ndarray, max_conditioning_vars: int = 3) -> float:
        """Test conditional independence of x and y given subsets of other variables"""
        
        # Simple correlation-based independence test
        base_correlation = abs(np.corrcoef(x, y)[0, 1])
        
        # Test conditioning on other variables
        other_vars = []
        for i in range(full_data.shape[1]):
            if not np.array_equal(full_data[:, i], x) and not np.array_equal(full_data[:, i], y):
                other_vars.append(i)
        
        if len(other_vars) == 0:
            return 1.0 - base_correlation
        
        # Test subsets of conditioning variables
        independence_scores = [1.0 - base_correlation]
        
        for subset_size in range(1, min(len(other_vars) + 1, max_conditioning_vars + 1)):
            for conditioning_vars in combinations(other_vars, subset_size):
                try:
                    # Partial correlation
                    partial_corr = self._partial_correlation(x, y, full_data[:, list(conditioning_vars)])
                    independence_scores.append(1.0 - abs(partial_corr))
                except:
                    continue
        
        # Return maximum independence score (most conservative)
        return max(independence_scores)
    
    def _partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Compute partial correlation of x and y given z"""
        if z.size == 0:
            return np.corrcoef(x, y)[0, 1]
        
        # Regression approach to partial correlation
        try:
            from sklearn.linear_model import LinearRegression
            
            # Regress x on z
            reg_x = LinearRegression().fit(z.reshape(-1, z.shape[1]) if z.ndim > 1 else z.reshape(-1, 1), x)
            x_residuals = x - reg_x.predict(z.reshape(-1, z.shape[1]) if z.ndim > 1 else z.reshape(-1, 1))
            
            # Regress y on z  
            reg_y = LinearRegression().fit(z.reshape(-1, z.shape[1]) if z.ndim > 1 else z.reshape(-1, 1), y)
            y_residuals = y - reg_y.predict(z.reshape(-1, z.shape[1]) if z.ndim > 1 else z.reshape(-1, 1))
            
            # Correlation of residuals
            return np.corrcoef(x_residuals, y_residuals)[0, 1]
            
        except ImportError:
            # Fallback: simple correlation if sklearn not available
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0.0
    
    def _compute_granger_causality(self, x: np.ndarray, y: np.ndarray, max_lags: int = 5) -> float:
        """Compute Granger causality from x to y"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            
            n = len(y)
            if n <= max_lags + 1:
                return 0.0
            
            # Create lagged features
            def create_lagged_features(series, lags):
                lagged = []
                for i in range(lags, len(series)):
                    lagged.append([series[i-j-1] for j in range(lags)])
                return np.array(lagged)
            
            # Model 1: y predicted by its own lags
            y_lagged = create_lagged_features(y, max_lags)
            y_target = y[max_lags:]
            
            reg1 = LinearRegression().fit(y_lagged, y_target)
            mse1 = mean_squared_error(y_target, reg1.predict(y_lagged))
            
            # Model 2: y predicted by its own lags + x lags  
            x_lagged = create_lagged_features(x, max_lags)
            min_len = min(len(y_lagged), len(x_lagged))
            
            combined_features = np.column_stack([y_lagged[:min_len], x_lagged[:min_len]])
            y_target_combined = y_target[:min_len]
            
            reg2 = LinearRegression().fit(combined_features, y_target_combined)
            mse2 = mean_squared_error(y_target_combined, reg2.predict(combined_features))
            
            # Granger causality strength
            if mse2 == 0:
                return 1.0 if mse1 > 0 else 0.0
            
            granger_strength = max(0, (mse1 - mse2) / mse1)
            return min(1.0, granger_strength)
            
        except ImportError:
            # Fallback: correlation-based measure
            logger.warning("sklearn not available, using correlation fallback for Granger causality")
            return abs(np.corrcoef(x[:-1], y[1:])[0, 1])
        except:
            return 0.0
    
    def _compute_information_geometric_causality(self, x: np.ndarray, y: np.ndarray, full_data: np.ndarray) -> float:
        """Novel information-geometric causality measure"""
        
        # Compute information-theoretic measures
        def entropy(data):
            """Estimate entropy using histogram method"""
            hist, _ = np.histogram(data, bins=min(50, len(data) // 10))
            hist = hist[hist > 0]  # Remove zero bins
            probs = hist / np.sum(hist)
            return -np.sum(probs * np.log2(probs))
        
        def mutual_information(x_data, y_data):
            """Estimate mutual information"""
            # 2D histogram
            hist_2d, _, _ = np.histogram2d(x_data, y_data, bins=min(20, len(x_data) // 20))
            hist_2d = hist_2d[hist_2d > 0]
            
            # Marginal histograms
            hist_x, _ = np.histogram(x_data, bins=min(20, len(x_data) // 20))
            hist_y, _ = np.histogram(y_data, bins=min(20, len(y_data) // 20))
            
            hist_x = hist_x[hist_x > 0]
            hist_y = hist_y[hist_y > 0]
            
            # Convert to probabilities
            p_joint = hist_2d / np.sum(hist_2d)
            p_x = hist_x / np.sum(hist_x)
            p_y = hist_y / np.sum(hist_y)
            
            # Mutual information
            mi = 0
            for p_xy in p_joint.flat:
                if p_xy > 0:
                    # This is a simplified approximation
                    mi += p_xy * np.log2(p_xy / (np.mean(p_x) * np.mean(p_y)))
            
            return max(0, mi)
        
        try:
            # Information geometric causality: I(X;Y|Past) where Past includes history
            mi_xy = mutual_information(x, y)
            
            # Condition on past values (simplified)
            if len(x) > 10:
                x_past = x[:-5]
                y_past = y[:-5]
                x_present = x[5:]
                y_present = y[5:]
                
                mi_conditional = mutual_information(x_present, y_present)
                
                # Geometric divergence measure (novel)
                geometric_causality = mi_conditional / (mi_xy + 1e-10)
                
                return min(1.0, max(0.0, geometric_causality))
            else:
                return min(1.0, mi_xy)
                
        except:
            # Fallback to simple correlation
            return abs(np.corrcoef(x, y)[0, 1])
    
    def _build_consensus(self, causal_votes: Dict[Tuple[str, str], List[CausalRelationship]], 
                        method_confidences: Dict[str, float]) -> List[CausalRelationship]:
        """Build consensus from multiple causal discovery methods"""
        
        final_edges = []
        
        for edge_key, votes in causal_votes.items():
            if len(votes) >= 2:  # Require at least 2 methods to agree
                # Weighted average of strengths
                total_weight = 0
                weighted_strength = 0
                weighted_confidence = 0
                all_evidence = {}
                
                for vote in votes:
                    method_name = vote.evidence.get('method', 'unknown')
                    weight = method_confidences.get(method_name, 1.0)
                    
                    weighted_strength += vote.strength * weight
                    weighted_confidence += vote.confidence * weight
                    total_weight += weight
                    
                    # Merge evidence
                    for k, v in vote.evidence.items():
                        all_evidence[k] = v
                
                if total_weight > 0:
                    consensus_strength = weighted_strength / total_weight
                    consensus_confidence = weighted_confidence / total_weight
                    
                    if consensus_strength >= self.min_causal_strength and consensus_confidence >= self.confidence_threshold:
                        final_edges.append(CausalRelationship(
                            cause=edge_key[0],
                            effect=edge_key[1],
                            strength=consensus_strength,
                            confidence=consensus_confidence,
                            mechanism="ensemble_consensus",
                            evidence=all_evidence
                        ))
        
        return final_edges
    
    def _apply_prior_knowledge(self, edges: List[CausalRelationship], 
                              prior_knowledge: Dict[str, Any]) -> List[CausalRelationship]:
        """Apply domain knowledge constraints to causal relationships"""
        
        filtered_edges = []
        
        # Forbidden edges
        forbidden = prior_knowledge.get('forbidden_edges', [])
        # Required edges  
        required = prior_knowledge.get('required_edges', [])
        # Temporal ordering
        temporal_order = prior_knowledge.get('temporal_order', {})
        
        for edge in edges:
            # Check forbidden edges
            if (edge.cause, edge.effect) in forbidden:
                logger.info(f"Filtered forbidden edge: {edge.cause} -> {edge.effect}")
                continue
            
            # Check temporal constraints
            cause_time = temporal_order.get(edge.cause, 0)
            effect_time = temporal_order.get(edge.effect, 1)
            
            if cause_time >= effect_time:  # Cause must precede effect
                logger.info(f"Filtered temporally invalid edge: {edge.cause} -> {edge.effect}")
                continue
            
            filtered_edges.append(edge)
        
        # Add required edges (if not already present)
        existing_edges = {(e.cause, e.effect) for e in filtered_edges}
        
        for cause, effect in required:
            if (cause, effect) not in existing_edges:
                filtered_edges.append(CausalRelationship(
                    cause=cause,
                    effect=effect,
                    strength=0.9,  # High strength for required edges
                    confidence=1.0,
                    mechanism="prior_knowledge", 
                    evidence={"source": "domain_expertise"}
                ))
        
        return filtered_edges
    
    def interventional_prediction(self, graph: CausalGraph, 
                                intervention: Dict[str, float],
                                target_variables: List[str]) -> Dict[str, float]:
        """Predict effects of interventions using causal graph"""
        
        logger.info(f"Predicting intervention effects: {intervention}")
        
        predictions = {}
        
        for target in target_variables:
            if target in intervention:
                # Direct intervention
                predictions[target] = intervention[target]
            else:
                # Compute causal effect through graph
                effect_strength = self._compute_causal_path_strength(
                    graph, list(intervention.keys()), target
                )
                
                if effect_strength > 0:
                    # Simple linear intervention model
                    intervention_magnitude = np.mean(list(intervention.values()))
                    predictions[target] = intervention_magnitude * effect_strength
                else:
                    predictions[target] = 0.0
        
        return predictions
    
    def _compute_causal_path_strength(self, graph: CausalGraph, 
                                    sources: List[str], target: str) -> float:
        """Compute total causal effect from sources to target through all paths"""
        
        total_effect = 0.0
        
        for source in sources:
            if source == target:
                continue
            
            # Find all directed paths from source to target
            paths = self._find_all_paths(graph, source, target)
            
            for path in paths:
                # Compute path strength (product of edge strengths)
                path_strength = 1.0
                for i in range(len(path) - 1):
                    edge_strength = self._get_edge_strength(graph, path[i], path[i+1])
                    if edge_strength == 0:
                        path_strength = 0
                        break
                    path_strength *= edge_strength
                
                total_effect += path_strength
        
        return min(1.0, total_effect)  # Cap at 1.0
    
    def _find_all_paths(self, graph: CausalGraph, start: str, end: str) -> List[List[str]]:
        """Find all directed paths from start to end in causal graph"""
        
        paths = []
        
        def dfs_path(current: str, target: str, path: List[str], visited: Set[str]):
            if current == target:
                paths.append(path.copy())
                return
            
            for child in graph.get_children(current):
                if child not in visited:  # Avoid cycles
                    visited.add(child)
                    path.append(child)
                    dfs_path(child, target, path, visited)
                    path.pop()
                    visited.remove(child)
        
        dfs_path(start, end, [start], {start})
        return paths
    
    def _get_edge_strength(self, graph: CausalGraph, cause: str, effect: str) -> float:
        """Get strength of specific causal edge"""
        
        for edge in graph.edges:
            if edge.cause == cause and edge.effect == effect:
                return edge.strength
        
        return 0.0
    
    def export_causal_graph(self, graph: CausalGraph, format: str = 'json') -> str:
        """Export causal graph in specified format"""
        
        if format == 'json':
            graph_dict = {
                'nodes': list(graph.nodes),
                'edges': [
                    {
                        'cause': edge.cause,
                        'effect': edge.effect,
                        'strength': edge.strength,
                        'confidence': edge.confidence,
                        'mechanism': edge.mechanism,
                        'evidence': edge.evidence
                    }
                    for edge in graph.edges
                ],
                'discovery_method': graph.discovery_method,
                'timestamp': graph.timestamp
            }
            return json.dumps(graph_dict, indent=2)
        
        elif format == 'dot':
            # GraphViz DOT format
            dot_content = "digraph CausalGraph {\n"
            
            # Add nodes
            for node in graph.nodes:
                dot_content += f'  "{node}" [shape=ellipse];\n'
            
            # Add edges
            for edge in graph.edges:
                dot_content += f'  "{edge.cause}" -> "{edge.effect}" '
                dot_content += f'[label="{edge.strength:.2f}" weight="{edge.strength}"];\n'
            
            dot_content += "}\n"
            return dot_content
        
        else:
            raise ValueError(f"Unsupported export format: {format}")