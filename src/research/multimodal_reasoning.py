"""Multi-Modal Scientific Reasoning Engine for Holistic Analysis"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from abc import ABC, abstractmethod

from ..utils.error_handling import robust_execution, DiscoveryError
from ..utils.validation import ValidationMixin

logger = logging.getLogger(__name__)


@dataclass
class ScientificEvidence:
    """Structured representation of scientific evidence"""
    evidence_id: str
    modality: str  # 'text', 'numerical', 'visual', 'temporal', 'spatial'
    content: Any  # The actual evidence data
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ScientificHypothesis:
    """Multi-modal scientific hypothesis"""
    hypothesis_id: str
    claim: str
    domain: str
    supporting_evidence: List[ScientificEvidence] = field(default_factory=list)
    contradicting_evidence: List[ScientificEvidence] = field(default_factory=list)
    confidence_score: float = 0.0
    novelty_score: float = 0.0
    testability_score: float = 0.0
    falsifiability_score: float = 0.0
    
    def overall_quality_score(self) -> float:
        """Compute overall hypothesis quality"""
        return (self.confidence_score * 0.3 + 
                self.novelty_score * 0.25 +
                self.testability_score * 0.25 + 
                self.falsifiability_score * 0.2)


@dataclass
class ReasoningResult:
    """Result from multi-modal reasoning process"""
    reasoning_type: str
    hypotheses: List[ScientificHypothesis]
    cross_modal_insights: List[str]
    confidence_assessment: Dict[str, float]
    novel_connections: List[Dict[str, Any]]
    reasoning_chain: List[str]
    quality_metrics: Dict[str, float]


class ModalityAnalyzer(ABC):
    """Abstract base for modality-specific analyzers"""
    
    @abstractmethod
    def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze data in specific modality"""
        pass
    
    @abstractmethod
    def extract_features(self, data: Any) -> np.ndarray:
        """Extract features for cross-modal analysis"""
        pass


class TextualAnalyzer(ModalityAnalyzer, ValidationMixin):
    """Advanced textual scientific content analyzer"""
    
    def __init__(self):
        self.scientific_terms = self._load_scientific_vocabulary()
        self.hypothesis_patterns = self._load_hypothesis_patterns()
    
    def analyze(self, text_data: Union[str, List[str]]) -> Dict[str, Any]:
        """Analyze scientific text content"""
        
        if isinstance(text_data, str):
            text_data = [text_data]
        
        analysis = {
            'key_concepts': [],
            'methodological_terms': [],
            'statistical_claims': [],
            'causal_language': [],
            'uncertainty_expressions': [],
            'novelty_indicators': [],
            'hypothesis_statements': []
        }
        
        for text in text_data:
            # Extract key scientific concepts
            concepts = self._extract_scientific_concepts(text)
            analysis['key_concepts'].extend(concepts)
            
            # Identify methodological approaches
            methods = self._identify_methodological_terms(text)
            analysis['methodological_terms'].extend(methods)
            
            # Find statistical claims
            stats = self._extract_statistical_claims(text)
            analysis['statistical_claims'].extend(stats)
            
            # Detect causal language
            causal = self._detect_causal_language(text)
            analysis['causal_language'].extend(causal)
            
            # Find uncertainty expressions
            uncertainty = self._extract_uncertainty_expressions(text)
            analysis['uncertainty_expressions'].extend(uncertainty)
            
            # Identify novelty claims
            novelty = self._identify_novelty_indicators(text)
            analysis['novelty_indicators'].extend(novelty)
            
            # Extract hypothesis statements
            hypotheses = self._extract_hypothesis_statements(text)
            analysis['hypothesis_statements'].extend(hypotheses)
        
        # Deduplicate and score
        for key in analysis:
            analysis[key] = list(set(analysis[key]))
        
        return analysis
    
    def extract_features(self, text_data: Union[str, List[str]]) -> np.ndarray:
        """Extract numerical features from text for cross-modal analysis"""
        
        analysis = self.analyze(text_data)
        
        # Create feature vector
        features = [
            len(analysis['key_concepts']),
            len(analysis['methodological_terms']),
            len(analysis['statistical_claims']),
            len(analysis['causal_language']),
            len(analysis['uncertainty_expressions']),
            len(analysis['novelty_indicators']),
            len(analysis['hypothesis_statements']),
            self._compute_semantic_complexity(text_data),
            self._compute_scientific_rigor_score(analysis),
            self._compute_innovation_score(analysis)
        ]
        
        return np.array(features, dtype=float)
    
    def _load_scientific_vocabulary(self) -> Dict[str, List[str]]:
        """Load domain-specific scientific vocabulary"""
        return {
            'physics': ['quantum', 'electromagnetic', 'thermodynamic', 'relativistic'],
            'biology': ['genetic', 'evolutionary', 'metabolic', 'cellular'],
            'chemistry': ['molecular', 'catalytic', 'stoichiometric', 'electrochemical'],
            'mathematics': ['algebraic', 'topological', 'stochastic', 'algorithmic'],
            'statistics': ['regression', 'correlation', 'significance', 'hypothesis']
        }
    
    def _load_hypothesis_patterns(self) -> List[str]:
        """Load patterns that indicate hypothesis statements"""
        return [
            'we hypothesize', 'we propose', 'we suggest', 'we conjecture',
            'it is likely that', 'we expect that', 'our model predicts',
            'the theory implies', 'we demonstrate that'
        ]
    
    def _extract_scientific_concepts(self, text: str) -> List[str]:
        """Extract key scientific concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        for domain, terms in self.scientific_terms.items():
            for term in terms:
                if term in text_lower:
                    concepts.append(f"{domain}:{term}")
        
        return concepts
    
    def _identify_methodological_terms(self, text: str) -> List[str]:
        """Identify methodological approaches in text"""
        method_terms = [
            'experimental', 'observational', 'computational', 'theoretical',
            'simulation', 'modeling', 'analysis', 'measurement', 'survey'
        ]
        
        found_methods = []
        text_lower = text.lower()
        
        for method in method_terms:
            if method in text_lower:
                found_methods.append(method)
        
        return found_methods
    
    def _extract_statistical_claims(self, text: str) -> List[str]:
        """Extract statistical claims and p-values"""
        import re
        
        claims = []
        
        # P-value patterns
        p_patterns = [
            r'p\s*[<>=]\s*0\.\d+',
            r'p-value\s*[<>=]\s*0\.\d+',
            r'significance\s*level',
            r'confidence\s*interval'
        ]
        
        for pattern in p_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend(matches)
        
        return claims
    
    def _detect_causal_language(self, text: str) -> List[str]:
        """Detect causal language patterns"""
        causal_terms = [
            'causes', 'leads to', 'results in', 'due to', 'because of',
            'triggers', 'induces', 'influences', 'affects', 'determines'
        ]
        
        found_causal = []
        text_lower = text.lower()
        
        for term in causal_terms:
            if term in text_lower:
                found_causal.append(term)
        
        return found_causal
    
    def _extract_uncertainty_expressions(self, text: str) -> List[str]:
        """Extract expressions of uncertainty"""
        uncertainty_terms = [
            'uncertain', 'unclear', 'ambiguous', 'tentative', 'preliminary',
            'may', 'might', 'could', 'possibly', 'potentially', 'likely'
        ]
        
        found_uncertainty = []
        text_lower = text.lower()
        
        for term in uncertainty_terms:
            if term in text_lower:
                found_uncertainty.append(term)
        
        return found_uncertainty
    
    def _identify_novelty_indicators(self, text: str) -> List[str]:
        """Identify claims of novelty or innovation"""
        novelty_terms = [
            'novel', 'new', 'innovative', 'unprecedented', 'breakthrough',
            'first time', 'pioneering', 'groundbreaking', 'revolutionary'
        ]
        
        found_novelty = []
        text_lower = text.lower()
        
        for term in novelty_terms:
            if term in text_lower:
                found_novelty.append(term)
        
        return found_novelty
    
    def _extract_hypothesis_statements(self, text: str) -> List[str]:
        """Extract explicit hypothesis statements"""
        hypotheses = []
        
        for pattern in self.hypothesis_patterns:
            if pattern in text.lower():
                # Extract sentence containing hypothesis
                sentences = text.split('.')
                for sentence in sentences:
                    if pattern in sentence.lower():
                        hypotheses.append(sentence.strip())
        
        return hypotheses
    
    def _compute_semantic_complexity(self, text_data: Union[str, List[str]]) -> float:
        """Compute semantic complexity score"""
        if isinstance(text_data, list):
            text = ' '.join(text_data)
        else:
            text = text_data
        
        # Simple complexity metrics
        word_count = len(text.split())
        unique_words = len(set(text.lower().split()))
        
        if word_count == 0:
            return 0.0
        
        complexity = unique_words / word_count
        return min(1.0, complexity)
    
    def _compute_scientific_rigor_score(self, analysis: Dict[str, Any]) -> float:
        """Compute scientific rigor score based on analysis"""
        rigor_indicators = [
            len(analysis['methodological_terms']) > 0,
            len(analysis['statistical_claims']) > 0,
            len(analysis['uncertainty_expressions']) > 0,
        ]
        
        return sum(rigor_indicators) / len(rigor_indicators)
    
    def _compute_innovation_score(self, analysis: Dict[str, Any]) -> float:
        """Compute innovation score"""
        innovation_score = len(analysis['novelty_indicators']) * 0.1
        return min(1.0, innovation_score)


class NumericalAnalyzer(ModalityAnalyzer, ValidationMixin):
    """Advanced numerical data analyzer for scientific insights"""
    
    def analyze(self, numerical_data: np.ndarray) -> Dict[str, Any]:
        """Analyze numerical scientific data"""
        
        if numerical_data.size == 0:
            return {'error': 'Empty data'}
        
        analysis = {
            'descriptive_stats': self._compute_descriptive_statistics(numerical_data),
            'distribution_properties': self._analyze_distributions(numerical_data),
            'correlation_structure': self._analyze_correlations(numerical_data),
            'anomaly_detection': self._detect_anomalies(numerical_data),
            'trend_analysis': self._analyze_trends(numerical_data),
            'statistical_tests': self._perform_statistical_tests(numerical_data)
        }
        
        return analysis
    
    def extract_features(self, numerical_data: np.ndarray) -> np.ndarray:
        """Extract meta-features from numerical data"""
        
        if numerical_data.size == 0:
            return np.zeros(10)
        
        # Ensure 2D
        if numerical_data.ndim == 1:
            numerical_data = numerical_data.reshape(-1, 1)
        
        features = []
        
        # Basic statistics
        features.append(np.mean(numerical_data))
        features.append(np.std(numerical_data))
        features.append(np.median(numerical_data))
        features.append(np.percentile(numerical_data, 75) - np.percentile(numerical_data, 25))  # IQR
        
        # Distribution properties
        features.append(self._compute_skewness(numerical_data.flatten()))
        features.append(self._compute_kurtosis(numerical_data.flatten()))
        
        # Complexity measures
        features.append(self._compute_entropy(numerical_data.flatten()))
        features.append(numerical_data.shape[1])  # Dimensionality
        features.append(numerical_data.shape[0])  # Sample size
        features.append(self._compute_correlation_strength(numerical_data))
        
        return np.array(features, dtype=float)
    
    def _compute_descriptive_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive descriptive statistics"""
        flat_data = data.flatten()
        
        return {
            'mean': float(np.mean(flat_data)),
            'median': float(np.median(flat_data)),
            'std': float(np.std(flat_data)),
            'var': float(np.var(flat_data)),
            'min': float(np.min(flat_data)),
            'max': float(np.max(flat_data)),
            'q25': float(np.percentile(flat_data, 25)),
            'q75': float(np.percentile(flat_data, 75)),
            'range': float(np.ptp(flat_data))
        }
    
    def _analyze_distributions(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze distributional properties"""
        flat_data = data.flatten()
        
        return {
            'skewness': self._compute_skewness(flat_data),
            'kurtosis': self._compute_kurtosis(flat_data),
            'entropy': self._compute_entropy(flat_data),
            'normality_test': self._test_normality(flat_data)
        }
    
    def _analyze_correlations(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze correlation structure"""
        if data.ndim == 1 or data.shape[1] == 1:
            return {'note': 'Single variable - no correlations'}
        
        corr_matrix = np.corrcoef(data.T)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        return {
            'mean_correlation': float(np.mean(np.abs(upper_triangle))),
            'max_correlation': float(np.max(np.abs(upper_triangle))),
            'correlation_matrix_condition': float(np.linalg.cond(corr_matrix))
        }
    
    def _detect_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in numerical data"""
        flat_data = data.flatten()
        
        # IQR method
        q1, q3 = np.percentile(flat_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_iqr = np.sum((flat_data < lower_bound) | (flat_data > upper_bound))
        
        # Z-score method
        z_scores = np.abs((flat_data - np.mean(flat_data)) / np.std(flat_data))
        outliers_zscore = np.sum(z_scores > 3)
        
        return {
            'outliers_iqr_method': int(outliers_iqr),
            'outliers_zscore_method': int(outliers_zscore),
            'outlier_percentage': float(outliers_iqr / len(flat_data) * 100)
        }
    
    def _analyze_trends(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal or sequential trends"""
        if data.ndim == 1:
            # Simple linear trend
            x = np.arange(len(data))
            trend_slope = np.polyfit(x, data, 1)[0]
            
            return {
                'linear_trend_slope': float(trend_slope),
                'trend_strength': float(abs(trend_slope))
            }
        else:
            # Multi-variate trend analysis
            trends = []
            for i in range(data.shape[1]):
                x = np.arange(len(data[:, i]))
                slope = np.polyfit(x, data[:, i], 1)[0]
                trends.append(slope)
            
            return {
                'mean_trend_slope': float(np.mean(trends)),
                'trend_variability': float(np.std(trends))
            }
    
    def _perform_statistical_tests(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform relevant statistical tests"""
        flat_data = data.flatten()
        
        tests = {}
        
        # Test for randomness (runs test approximation)
        median_val = np.median(flat_data)
        runs = 1
        for i in range(1, len(flat_data)):
            if (flat_data[i] >= median_val) != (flat_data[i-1] >= median_val):
                runs += 1
        
        expected_runs = (2 * len(flat_data) + 1) / 3
        tests['runs_test_statistic'] = float(runs)
        tests['randomness_score'] = float(abs(runs - expected_runs) / expected_runs)
        
        return tests
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness"""
        n = len(data)
        if n < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.sum(((data - mean_val) / std_val) ** 3) / n
        return float(skewness)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis"""
        n = len(data)
        if n < 4:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = np.sum(((data - mean_val) / std_val) ** 4) / n - 3
        return float(kurtosis)
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy of data distribution"""
        if len(data) == 0:
            return 0.0
        
        # Discretize data for entropy calculation
        bins = min(50, len(data) // 10) if len(data) > 10 else len(data)
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        probs = hist / np.sum(hist)
        entropy = -np.sum(probs * np.log2(probs))
        
        return float(entropy)
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, float]:
        """Simple normality test"""
        if len(data) < 3:
            return {'normality_score': 0.5}
        
        # Simple normality check based on skewness and kurtosis
        skew = abs(self._compute_skewness(data))
        kurt = abs(self._compute_kurtosis(data))
        
        # Rough normality score (lower is more normal)
        normality_score = 1.0 - min(1.0, (skew + kurt) / 4.0)
        
        return {'normality_score': float(normality_score)}
    
    def _compute_correlation_strength(self, data: np.ndarray) -> float:
        """Compute overall correlation strength"""
        if data.ndim == 1 or data.shape[1] == 1:
            return 0.0
        
        try:
            corr_matrix = np.corrcoef(data.T)
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            return float(np.mean(np.abs(upper_triangle)))
        except:
            return 0.0


class MultiModalReasoningEngine(ValidationMixin):
    """Advanced multi-modal scientific reasoning system"""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 novelty_threshold: float = 0.5):
        """
        Initialize multi-modal reasoning engine
        
        Args:
            confidence_threshold: Minimum confidence for accepting hypotheses
            novelty_threshold: Minimum novelty score for interesting hypotheses
        """
        self.confidence_threshold = confidence_threshold
        self.novelty_threshold = novelty_threshold
        
        # Initialize modality analyzers
        self.text_analyzer = TextualAnalyzer()
        self.numerical_analyzer = NumericalAnalyzer()
        
        # Reasoning history
        self.reasoning_history: List[ReasoningResult] = []
        
        logger.info("MultiModalReasoningEngine initialized")
    
    @robust_execution(recovery_strategy='partial_recovery')
    def holistic_scientific_reasoning(self,
                                    text_data: Optional[Union[str, List[str]]] = None,
                                    numerical_data: Optional[np.ndarray] = None,
                                    domain: str = "general_science",
                                    prior_hypotheses: Optional[List[ScientificHypothesis]] = None) -> ReasoningResult:
        """
        Perform holistic multi-modal scientific reasoning
        
        Args:
            text_data: Scientific text content
            numerical_data: Numerical data
            domain: Scientific domain
            prior_hypotheses: Existing hypotheses to build upon
            
        Returns:
            ReasoningResult with generated hypotheses and insights
        """
        
        logger.info(f"Starting holistic reasoning in domain: {domain}")
        
        # Step 1: Analyze each modality independently
        modality_analyses = {}
        
        if text_data is not None:
            modality_analyses['text'] = self.text_analyzer.analyze(text_data)
            
        if numerical_data is not None:
            modality_analyses['numerical'] = self.numerical_analyzer.analyze(numerical_data)
        
        # Step 2: Extract cross-modal features
        cross_modal_features = self._extract_cross_modal_features(
            text_data, numerical_data
        )
        
        # Step 3: Generate hypotheses from each modality
        hypotheses = []
        
        if text_data is not None:
            text_hypotheses = self._generate_text_based_hypotheses(
                modality_analyses['text'], domain
            )
            hypotheses.extend(text_hypotheses)
        
        if numerical_data is not None:
            numerical_hypotheses = self._generate_numerical_hypotheses(
                modality_analyses['numerical'], domain
            )
            hypotheses.extend(numerical_hypotheses)
        
        # Step 4: Cross-modal hypothesis synthesis
        synthesized_hypotheses = self._synthesize_cross_modal_hypotheses(
            hypotheses, cross_modal_features, domain
        )
        hypotheses.extend(synthesized_hypotheses)
        
        # Step 5: Hypothesis validation and ranking
        validated_hypotheses = self._validate_and_rank_hypotheses(
            hypotheses, modality_analyses
        )
        
        # Step 6: Generate cross-modal insights
        cross_modal_insights = self._generate_cross_modal_insights(
            modality_analyses, cross_modal_features
        )
        
        # Step 7: Identify novel connections
        novel_connections = self._identify_novel_connections(
            validated_hypotheses, modality_analyses
        )
        
        # Step 8: Build reasoning chain
        reasoning_chain = self._construct_reasoning_chain(
            modality_analyses, validated_hypotheses, cross_modal_insights
        )
        
        # Step 9: Assess overall confidence
        confidence_assessment = self._assess_reasoning_confidence(
            modality_analyses, validated_hypotheses
        )
        
        # Step 10: Compute quality metrics
        quality_metrics = self._compute_reasoning_quality_metrics(
            validated_hypotheses, cross_modal_insights, novel_connections
        )
        
        result = ReasoningResult(
            reasoning_type="holistic_multimodal",
            hypotheses=validated_hypotheses,
            cross_modal_insights=cross_modal_insights,
            confidence_assessment=confidence_assessment,
            novel_connections=novel_connections,
            reasoning_chain=reasoning_chain,
            quality_metrics=quality_metrics
        )
        
        self.reasoning_history.append(result)
        
        logger.info(f"Generated {len(validated_hypotheses)} validated hypotheses with "
                   f"{len(cross_modal_insights)} cross-modal insights")
        
        return result
    
    def _extract_cross_modal_features(self,
                                    text_data: Optional[Union[str, List[str]]],
                                    numerical_data: Optional[np.ndarray]) -> np.ndarray:
        """Extract features that bridge different modalities"""
        
        features = []
        
        # Text features
        if text_data is not None:
            text_features = self.text_analyzer.extract_features(text_data)
            features.extend(text_features)
        else:
            features.extend([0.0] * 10)  # Placeholder for missing text
        
        # Numerical features
        if numerical_data is not None:
            num_features = self.numerical_analyzer.extract_features(numerical_data)
            features.extend(num_features)
        else:
            features.extend([0.0] * 10)  # Placeholder for missing numerical
        
        # Cross-modal alignment features
        if text_data is not None and numerical_data is not None:
            alignment_features = self._compute_alignment_features(text_data, numerical_data)
            features.extend(alignment_features)
        else:
            features.extend([0.0] * 5)  # Placeholder for missing alignment
        
        return np.array(features, dtype=float)
    
    def _compute_alignment_features(self,
                                  text_data: Union[str, List[str]],
                                  numerical_data: np.ndarray) -> List[float]:
        """Compute features that measure alignment between text and numerical data"""
        
        text_analysis = self.text_analyzer.analyze(text_data)
        numerical_analysis = self.numerical_analyzer.analyze(numerical_data)
        
        alignment_features = []
        
        # Statistical alignment
        has_statistical_claims = len(text_analysis.get('statistical_claims', [])) > 0
        has_numerical_anomalies = numerical_analysis.get('anomaly_detection', {}).get('outlier_percentage', 0) > 5
        alignment_features.append(float(has_statistical_claims and has_numerical_anomalies))
        
        # Uncertainty alignment
        has_uncertainty_text = len(text_analysis.get('uncertainty_expressions', [])) > 0
        has_high_variance = numerical_analysis.get('descriptive_stats', {}).get('std', 0) > 1.0
        alignment_features.append(float(has_uncertainty_text and has_high_variance))
        
        # Trend alignment
        has_causal_language = len(text_analysis.get('causal_language', [])) > 0
        has_trends = abs(numerical_analysis.get('trend_analysis', {}).get('linear_trend_slope', 0)) > 0.1
        alignment_features.append(float(has_causal_language and has_trends))
        
        # Innovation alignment
        has_novelty_claims = len(text_analysis.get('novelty_indicators', [])) > 0
        has_unusual_distribution = numerical_analysis.get('distribution_properties', {}).get('normality_test', {}).get('normality_score', 1.0) < 0.3
        alignment_features.append(float(has_novelty_claims and has_unusual_distribution))
        
        # Overall coherence
        text_complexity = len(text_analysis.get('key_concepts', [])) / 10.0
        numerical_complexity = numerical_analysis.get('descriptive_stats', {}).get('std', 0) / np.maximum(1.0, abs(numerical_analysis.get('descriptive_stats', {}).get('mean', 1.0)))
        coherence = 1.0 - abs(text_complexity - numerical_complexity)
        alignment_features.append(float(coherence))
        
        return alignment_features
    
    def _generate_text_based_hypotheses(self,
                                      text_analysis: Dict[str, Any],
                                      domain: str) -> List[ScientificHypothesis]:
        """Generate hypotheses from textual analysis"""
        
        hypotheses = []
        
        # Hypothesis from explicit statements
        for i, statement in enumerate(text_analysis.get('hypothesis_statements', [])):
            hypothesis = ScientificHypothesis(
                hypothesis_id=f"text_explicit_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                claim=statement,
                domain=domain,
                confidence_score=0.8,  # High confidence for explicit statements
                testability_score=0.7,
                falsifiability_score=0.6
            )
            hypotheses.append(hypothesis)
        
        # Hypothesis from causal language
        if text_analysis.get('causal_language'):
            causal_claim = f"Causal relationships exist in {domain} involving: {', '.join(text_analysis['causal_language'][:3])}"
            hypothesis = ScientificHypothesis(
                hypothesis_id=f"text_causal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                claim=causal_claim,
                domain=domain,
                confidence_score=0.6,
                testability_score=0.8,
                falsifiability_score=0.7
            )
            hypotheses.append(hypothesis)
        
        # Hypothesis from novelty indicators
        if text_analysis.get('novelty_indicators'):
            novelty_claim = f"Novel phenomena in {domain} characterized by: {', '.join(text_analysis['novelty_indicators'][:3])}"
            hypothesis = ScientificHypothesis(
                hypothesis_id=f"text_novelty_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                claim=novelty_claim,
                domain=domain,
                confidence_score=0.5,  # Lower confidence for novelty claims
                novelty_score=0.9,  # High novelty by definition
                testability_score=0.6,
                falsifiability_score=0.5
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_numerical_hypotheses(self,
                                     numerical_analysis: Dict[str, Any],
                                     domain: str) -> List[ScientificHypothesis]:
        """Generate hypotheses from numerical data analysis"""
        
        hypotheses = []
        
        # Trend-based hypothesis
        trend_info = numerical_analysis.get('trend_analysis', {})
        if abs(trend_info.get('linear_trend_slope', 0)) > 0.1:
            trend_direction = "increasing" if trend_info['linear_trend_slope'] > 0 else "decreasing"
            claim = f"Data exhibits {trend_direction} trend in {domain} with slope {trend_info['linear_trend_slope']:.3f}"
            
            hypothesis = ScientificHypothesis(
                hypothesis_id=f"numerical_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                claim=claim,
                domain=domain,
                confidence_score=min(0.9, abs(trend_info['linear_trend_slope']) * 2),
                testability_score=0.9,  # Trends are highly testable
                falsifiability_score=0.8
            )
            hypotheses.append(hypothesis)
        
        # Distribution-based hypothesis
        dist_info = numerical_analysis.get('distribution_properties', {})
        normality_score = dist_info.get('normality_test', {}).get('normality_score', 0.5)
        
        if normality_score < 0.3:  # Non-normal distribution
            claim = f"Non-normal distribution in {domain} suggests underlying complex processes (normality score: {normality_score:.3f})"
            hypothesis = ScientificHypothesis(
                hypothesis_id=f"numerical_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                claim=claim,
                domain=domain,
                confidence_score=1.0 - normality_score,
                novelty_score=0.7,
                testability_score=0.8,
                falsifiability_score=0.7
            )
            hypotheses.append(hypothesis)
        
        # Anomaly-based hypothesis
        anomaly_info = numerical_analysis.get('anomaly_detection', {})
        outlier_percentage = anomaly_info.get('outlier_percentage', 0)
        
        if outlier_percentage > 10:  # Significant outliers
            claim = f"Significant anomalies ({outlier_percentage:.1f}%) in {domain} data suggest rare events or measurement errors"
            hypothesis = ScientificHypothesis(
                hypothesis_id=f"numerical_anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                claim=claim,
                domain=domain,
                confidence_score=min(0.9, outlier_percentage / 20),
                testability_score=0.8,
                falsifiability_score=0.9
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _synthesize_cross_modal_hypotheses(self,
                                         existing_hypotheses: List[ScientificHypothesis],
                                         cross_modal_features: np.ndarray,
                                         domain: str) -> List[ScientificHypothesis]:
        """Synthesize novel hypotheses by combining insights from multiple modalities"""
        
        synthesized = []
        
        # Check if we have sufficient cross-modal alignment
        if len(cross_modal_features) >= 25:  # Ensure we have all feature types
            text_features = cross_modal_features[:10]
            numerical_features = cross_modal_features[10:20]
            alignment_features = cross_modal_features[20:25]
            
            # High alignment suggests coherent multi-modal evidence
            alignment_strength = np.mean(alignment_features)
            
            if alignment_strength > 0.6:
                claim = f"Multi-modal evidence suggests coherent underlying mechanisms in {domain}"
                hypothesis = ScientificHypothesis(
                    hypothesis_id=f"crossmodal_coherence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    claim=claim,
                    domain=domain,
                    confidence_score=alignment_strength,
                    novelty_score=0.8,  # Cross-modal synthesis is inherently novel
                    testability_score=0.7,
                    falsifiability_score=0.6
                )
                synthesized.append(hypothesis)
            
            # Contradiction detection
            if alignment_strength < 0.3:
                claim = f"Multi-modal inconsistencies in {domain} suggest measurement issues or complex underlying dynamics"
                hypothesis = ScientificHypothesis(
                    hypothesis_id=f"crossmodal_contradiction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    claim=claim,
                    domain=domain,
                    confidence_score=1.0 - alignment_strength,
                    novelty_score=0.6,
                    testability_score=0.8,
                    falsifiability_score=0.9
                )
                synthesized.append(hypothesis)
        
        return synthesized
    
    def _validate_and_rank_hypotheses(self,
                                    hypotheses: List[ScientificHypothesis],
                                    modality_analyses: Dict[str, Dict[str, Any]]) -> List[ScientificHypothesis]:
        """Validate and rank hypotheses by quality"""
        
        validated = []
        
        for hypothesis in hypotheses:
            # Update confidence based on supporting evidence
            self._update_hypothesis_confidence(hypothesis, modality_analyses)
            
            # Compute overall quality score
            quality_score = hypothesis.overall_quality_score()
            
            # Filter by thresholds
            if (hypothesis.confidence_score >= self.confidence_threshold or
                hypothesis.novelty_score >= self.novelty_threshold):
                validated.append(hypothesis)
        
        # Sort by overall quality score
        validated.sort(key=lambda h: h.overall_quality_score(), reverse=True)
        
        return validated
    
    def _update_hypothesis_confidence(self,
                                    hypothesis: ScientificHypothesis,
                                    modality_analyses: Dict[str, Dict[str, Any]]):
        """Update hypothesis confidence based on supporting evidence"""
        
        evidence_count = 0
        total_confidence = hypothesis.confidence_score
        
        # Check for supporting evidence in text analysis
        if 'text' in modality_analyses:
            text_analysis = modality_analyses['text']
            
            # Evidence from statistical claims
            if text_analysis.get('statistical_claims') and 'statistical' in hypothesis.claim.lower():
                evidence_count += 1
                total_confidence += 0.1
            
            # Evidence from causal language
            if text_analysis.get('causal_language') and 'causal' in hypothesis.claim.lower():
                evidence_count += 1
                total_confidence += 0.1
            
            # Evidence from uncertainty expressions
            if text_analysis.get('uncertainty_expressions') and ('uncertain' in hypothesis.claim.lower() or 'suggest' in hypothesis.claim.lower()):
                evidence_count += 1
                total_confidence += 0.05
        
        # Check for supporting evidence in numerical analysis
        if 'numerical' in modality_analyses:
            numerical_analysis = modality_analyses['numerical']
            
            # Evidence from trends
            trend_slope = abs(numerical_analysis.get('trend_analysis', {}).get('linear_trend_slope', 0))
            if trend_slope > 0.1 and ('trend' in hypothesis.claim.lower() or 'increasing' in hypothesis.claim.lower() or 'decreasing' in hypothesis.claim.lower()):
                evidence_count += 1
                total_confidence += min(0.2, trend_slope)
            
            # Evidence from anomalies
            outlier_percentage = numerical_analysis.get('anomaly_detection', {}).get('outlier_percentage', 0)
            if outlier_percentage > 10 and ('anomal' in hypothesis.claim.lower() or 'outlier' in hypothesis.claim.lower()):
                evidence_count += 1
                total_confidence += min(0.15, outlier_percentage / 100)
        
        # Update confidence (cap at 1.0)
        hypothesis.confidence_score = min(1.0, total_confidence)
    
    def _generate_cross_modal_insights(self,
                                     modality_analyses: Dict[str, Dict[str, Any]],
                                     cross_modal_features: np.ndarray) -> List[str]:
        """Generate insights from cross-modal analysis"""
        
        insights = []
        
        # Text-Numerical alignment insights
        if 'text' in modality_analyses and 'numerical' in modality_analyses:
            text_analysis = modality_analyses['text']
            numerical_analysis = modality_analyses['numerical']
            
            # Statistical claims vs actual statistics
            has_statistical_claims = len(text_analysis.get('statistical_claims', [])) > 0
            has_significant_trends = abs(numerical_analysis.get('trend_analysis', {}).get('linear_trend_slope', 0)) > 0.1
            
            if has_statistical_claims and has_significant_trends:
                insights.append("Text statistical claims are supported by observed numerical trends")
            elif has_statistical_claims and not has_significant_trends:
                insights.append("Text statistical claims are not clearly supported by numerical evidence")
            
            # Uncertainty vs variability
            has_uncertainty_language = len(text_analysis.get('uncertainty_expressions', [])) > 0
            high_variability = numerical_analysis.get('descriptive_stats', {}).get('std', 0) > 1.0
            
            if has_uncertainty_language and high_variability:
                insights.append("Textual uncertainty expressions align with high numerical variability")
            elif has_uncertainty_language and not high_variability:
                insights.append("Textual uncertainty not reflected in numerical stability")
            
            # Novelty vs unusual patterns
            has_novelty_claims = len(text_analysis.get('novelty_indicators', [])) > 0
            unusual_distribution = numerical_analysis.get('distribution_properties', {}).get('normality_test', {}).get('normality_score', 1.0) < 0.3
            
            if has_novelty_claims and unusual_distribution:
                insights.append("Novelty claims supported by unusual data distribution patterns")
        
        # Cross-modal feature insights
        if len(cross_modal_features) >= 25:
            alignment_strength = np.mean(cross_modal_features[20:25])
            
            if alignment_strength > 0.8:
                insights.append("Strong cross-modal coherence suggests robust underlying phenomena")
            elif alignment_strength < 0.3:
                insights.append("Low cross-modal coherence indicates potential inconsistencies or complex dynamics")
        
        return insights
    
    def _identify_novel_connections(self,
                                  hypotheses: List[ScientificHypothesis],
                                  modality_analyses: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify novel connections between different aspects of the analysis"""
        
        connections = []
        
        # Connection between high-novelty hypotheses
        novel_hypotheses = [h for h in hypotheses if h.novelty_score > 0.7]
        
        if len(novel_hypotheses) >= 2:
            for i, h1 in enumerate(novel_hypotheses):
                for h2 in novel_hypotheses[i+1:]:
                    # Check for conceptual overlap
                    overlap_score = self._compute_conceptual_overlap(h1.claim, h2.claim)
                    
                    if overlap_score > 0.3:
                        connections.append({
                            'type': 'hypothesis_convergence',
                            'hypothesis_1': h1.hypothesis_id,
                            'hypothesis_2': h2.hypothesis_id,
                            'connection_strength': overlap_score,
                            'description': f"Convergent evidence from multiple sources supporting related claims"
                        })
        
        # Connection between modality-specific insights
        if 'text' in modality_analyses and 'numerical' in modality_analyses:
            text_concepts = modality_analyses['text'].get('key_concepts', [])
            numerical_anomalies = modality_analyses['numerical'].get('anomaly_detection', {}).get('outlier_percentage', 0)
            
            if len(text_concepts) > 3 and numerical_anomalies > 15:
                connections.append({
                    'type': 'complexity_anomaly_connection',
                    'description': f"High conceptual complexity ({len(text_concepts)} concepts) correlates with data anomalies ({numerical_anomalies:.1f}%)"
                })
        
        return connections
    
    def _construct_reasoning_chain(self,
                                 modality_analyses: Dict[str, Dict[str, Any]],
                                 hypotheses: List[ScientificHypothesis],
                                 insights: List[str]) -> List[str]:
        """Construct logical reasoning chain"""
        
        chain = ["Multi-modal scientific reasoning initiated"]
        
        # Analysis steps
        if 'text' in modality_analyses:
            key_concepts = len(modality_analyses['text'].get('key_concepts', []))
            chain.append(f"Text analysis identified {key_concepts} key concepts")
        
        if 'numerical' in modality_analyses:
            trend_strength = abs(modality_analyses['numerical'].get('trend_analysis', {}).get('linear_trend_slope', 0))
            chain.append(f"Numerical analysis revealed trend strength: {trend_strength:.3f}")
        
        # Hypothesis generation
        chain.append(f"Generated {len(hypotheses)} scientific hypotheses")
        
        # Validation
        high_confidence = len([h for h in hypotheses if h.confidence_score > 0.8])
        chain.append(f"Validated {high_confidence} high-confidence hypotheses")
        
        # Cross-modal insights
        chain.append(f"Identified {len(insights)} cross-modal insights")
        
        # Final assessment
        if hypotheses:
            best_hypothesis = max(hypotheses, key=lambda h: h.overall_quality_score())
            chain.append(f"Best hypothesis: {best_hypothesis.claim[:100]}...")
        
        return chain
    
    def _assess_reasoning_confidence(self,
                                   modality_analyses: Dict[str, Dict[str, Any]],
                                   hypotheses: List[ScientificHypothesis]) -> Dict[str, float]:
        """Assess overall confidence in reasoning process"""
        
        assessment = {}
        
        # Data quality assessment
        if 'text' in modality_analyses:
            text_quality = len(modality_analyses['text'].get('key_concepts', [])) / 10.0
            assessment['text_data_quality'] = min(1.0, text_quality)
        
        if 'numerical' in modality_analyses:
            # Simple quality measure based on data characteristics
            stats = modality_analyses['numerical'].get('descriptive_stats', {})
            numerical_quality = 1.0 - min(1.0, stats.get('std', 0) / max(0.1, abs(stats.get('mean', 1.0))))
            assessment['numerical_data_quality'] = max(0.0, numerical_quality)
        
        # Hypothesis quality
        if hypotheses:
            avg_confidence = np.mean([h.confidence_score for h in hypotheses])
            avg_quality = np.mean([h.overall_quality_score() for h in hypotheses])
            assessment['average_hypothesis_confidence'] = avg_confidence
            assessment['average_hypothesis_quality'] = avg_quality
        
        # Overall reasoning confidence
        quality_scores = list(assessment.values())
        assessment['overall_reasoning_confidence'] = np.mean(quality_scores) if quality_scores else 0.0
        
        return assessment
    
    def _compute_reasoning_quality_metrics(self,
                                         hypotheses: List[ScientificHypothesis],
                                         insights: List[str],
                                         connections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute metrics for reasoning quality"""
        
        metrics = {}
        
        # Hypothesis diversity
        if hypotheses:
            domains = set(h.domain for h in hypotheses)
            mechanisms = set(h.mechanism if hasattr(h, 'mechanism') else 'unknown' for h in hypotheses)
            metrics['hypothesis_diversity'] = (len(domains) + len(mechanisms)) / (2 * len(hypotheses))
        
        # Insight density
        total_content = len(hypotheses) + len(insights) + len(connections)
        if total_content > 0:
            metrics['insight_density'] = len(insights) / total_content
        
        # Novelty vs confidence balance
        if hypotheses:
            novelty_scores = [h.novelty_score for h in hypotheses]
            confidence_scores = [h.confidence_score for h in hypotheses]
            
            metrics['novelty_confidence_balance'] = 1.0 - abs(np.mean(novelty_scores) - np.mean(confidence_scores))
        
        # Connection richness
        metrics['connection_richness'] = len(connections) / max(1, len(hypotheses))
        
        return metrics
    
    def _compute_conceptual_overlap(self, claim1: str, claim2: str) -> float:
        """Compute conceptual overlap between two claims"""
        
        # Simple word-based overlap
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())
        
        # Remove common words
        common_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words1 = words1 - common_stop_words
        words2 = words2 - common_stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0