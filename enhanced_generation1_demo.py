"""Enhanced Generation 1 Demo: Advanced Scientific Discovery Capabilities"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Demonstrate enhanced Generation 1 capabilities"""
    
    print("🚀 TERRAGON LABS - ENHANCED GENERATION 1 SCIENTIFIC DISCOVERY PLATFORM")
    print("=" * 80)
    print()
    
    # Generate comprehensive scientific dataset
    print("📊 GENERATING MULTI-MODAL SCIENTIFIC DATASET...")
    data_package = generate_scientific_data()
    
    print("🧠 DEMONSTRATION 1: ADVANCED CAUSAL DISCOVERY")
    print("-" * 50)
    demonstrate_causal_discovery(data_package['causal_data'], data_package['variable_names'])
    
    print("\n🔬 DEMONSTRATION 2: MULTI-MODAL SCIENTIFIC REASONING")
    print("-" * 50)  
    demonstrate_multimodal_reasoning(data_package['text_data'], data_package['numerical_data'])
    
    print("\n🧪 DEMONSTRATION 3: AUTONOMOUS HYPOTHESIS VALIDATION")
    print("-" * 50)
    demonstrate_hypothesis_validation(data_package['validation_data'])
    
    print("\n🎯 DEMONSTRATION 4: INTEGRATED SCIENTIFIC PIPELINE")
    print("-" * 50)
    demonstrate_integrated_pipeline(data_package)
    
    print("\n✅ ENHANCED GENERATION 1 DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("🏆 KEY ACHIEVEMENTS:")
    print("• Advanced causal discovery with multi-method ensemble")
    print("• Multi-modal reasoning combining text and numerical evidence")
    print("• Autonomous hypothesis design and validation")
    print("• Integrated scientific discovery pipeline")
    print("• Novel algorithms for breakthrough research")


def generate_scientific_data():
    """Generate comprehensive multi-modal scientific dataset"""
    
    np.random.seed(42)  # Reproducible results
    
    print("   • Generating causal relationships...")
    # Causal data: X -> Y -> Z with confounders
    n_samples = 500
    noise_level = 0.3
    
    # Causal chain: Temperature -> Pressure -> Density
    temperature = np.random.normal(300, 50, n_samples)  # Kelvin
    pressure = 0.8 * temperature + np.random.normal(0, noise_level * 50, n_samples)
    density = -0.6 * pressure + 0.4 * temperature + np.random.normal(0, noise_level * 20, n_samples)
    
    # Additional variables
    humidity = np.random.normal(60, 15, n_samples)  # Independent
    measurement_error = np.random.normal(0, 2, n_samples)
    
    causal_data = np.column_stack([
        temperature, pressure, density, humidity, measurement_error
    ])
    variable_names = ['Temperature', 'Pressure', 'Density', 'Humidity', 'MeasurementError']
    
    print("   • Generating numerical analysis data...")
    # Complex multi-dimensional data with trends and anomalies
    time_points = np.linspace(0, 10, n_samples)
    trend_data = 2.5 * time_points + 0.1 * time_points**2
    seasonal_data = 3 * np.sin(2 * np.pi * time_points)
    noise_data = np.random.normal(0, 0.5, n_samples)
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    noise_data[outlier_indices] += np.random.normal(0, 5, len(outlier_indices))
    
    numerical_data = trend_data + seasonal_data + noise_data
    
    print("   • Generating scientific text data...")
    # Scientific literature snippets
    text_data = [
        "We hypothesize that increased temperature leads to higher pressure in closed systems according to Gay-Lussac's law.",
        "Statistical analysis shows significant correlation (p < 0.001) between temperature and pressure measurements.",
        "Novel findings suggest that density variations may be influenced by both temperature and pressure through complex thermodynamic processes.",
        "The experimental methodology employed rigorous controls to minimize measurement errors and confounding variables.",
        "Uncertainty in measurements remains a significant challenge, requiring innovative approaches to error quantification.",
        "Breakthrough results indicate previously unknown relationships between environmental factors and system behavior.",
        "Causal mechanisms underlying the observed phenomena require further investigation through controlled experiments."
    ]
    
    print("   • Generating validation datasets...")
    # Validation data with known properties for testing
    validation_data = np.random.normal(10, 3, 200)  # Should reject null hypothesis that mean = 0
    
    return {
        'causal_data': causal_data,
        'variable_names': variable_names,
        'numerical_data': numerical_data.reshape(-1, 1),
        'text_data': text_data,
        'validation_data': validation_data,
        'time_points': time_points
    }


def demonstrate_causal_discovery(causal_data, variable_names):
    """Demonstrate advanced causal discovery capabilities"""
    
    try:
        from src.algorithms.causal_discovery import CausalDiscoveryEngine
        
        print("   • Initializing Causal Discovery Engine...")
        causal_engine = CausalDiscoveryEngine(
            min_causal_strength=0.3,
            confidence_threshold=0.6
        )
        
        print("   • Discovering causal structure using ensemble methods...")
        causal_graph = causal_engine.discover_causal_structure(
            data=causal_data,
            variable_names=variable_names,
            methods=['pc_algorithm', 'granger_causality', 'information_geometric'],
            prior_knowledge={
                'temporal_order': {
                    'Temperature': 0, 'Pressure': 1, 'Density': 2, 
                    'Humidity': 0, 'MeasurementError': 3
                },
                'forbidden_edges': [('Density', 'Temperature')]  # No reverse causation
            }
        )
        
        print(f"   ✅ Discovered {len(causal_graph.edges)} causal relationships:")
        for edge in causal_graph.edges:
            print(f"      • {edge.cause} → {edge.effect} "
                  f"(strength: {edge.strength:.3f}, confidence: {edge.confidence:.3f})")
        
        # Test interventional prediction
        print("   • Testing interventional predictions...")
        intervention = {'Temperature': 350}  # Increase temperature by 50K
        predictions = causal_engine.interventional_prediction(
            causal_graph, intervention, ['Pressure', 'Density']
        )
        
        print("   📊 Predicted intervention effects:")
        for var, effect in predictions.items():
            print(f"      • {var}: {effect:+.2f}")
        
        # Export causal graph
        graph_json = causal_engine.export_causal_graph(causal_graph, format='json')
        print("   💾 Causal graph exported to JSON format")
        
    except Exception as e:
        print(f"   ❌ Causal discovery failed: {e}")


def demonstrate_multimodal_reasoning(text_data, numerical_data):
    """Demonstrate multi-modal scientific reasoning"""
    
    try:
        from src.research.multimodal_reasoning import MultiModalReasoningEngine
        
        print("   • Initializing Multi-Modal Reasoning Engine...")
        reasoning_engine = MultiModalReasoningEngine(
            confidence_threshold=0.6,
            novelty_threshold=0.4
        )
        
        print("   • Performing holistic multi-modal analysis...")
        reasoning_result = reasoning_engine.holistic_scientific_reasoning(
            text_data=text_data,
            numerical_data=numerical_data,
            domain="thermodynamics"
        )
        
        print(f"   ✅ Generated {len(reasoning_result.hypotheses)} scientific hypotheses:")
        for i, hypothesis in enumerate(reasoning_result.hypotheses[:3], 1):  # Show top 3
            print(f"      {i}. {hypothesis.claim}")
            print(f"         • Confidence: {hypothesis.confidence_score:.3f}")
            print(f"         • Novelty: {hypothesis.novelty_score:.3f}")
            print(f"         • Quality: {hypothesis.overall_quality_score():.3f}")
        
        print(f"   🔍 Cross-modal insights ({len(reasoning_result.cross_modal_insights)}):")
        for insight in reasoning_result.cross_modal_insights:
            print(f"      • {insight}")
        
        print(f"   🔗 Novel connections ({len(reasoning_result.novel_connections)}):")
        for connection in reasoning_result.novel_connections:
            print(f"      • {connection.get('description', 'Unknown connection')}")
        
        print("   📈 Reasoning quality metrics:")
        for metric, value in reasoning_result.quality_metrics.items():
            print(f"      • {metric}: {value:.3f}")
        
    except Exception as e:
        print(f"   ❌ Multi-modal reasoning failed: {e}")


def demonstrate_hypothesis_validation(validation_data):
    """Demonstrate autonomous hypothesis validation"""
    
    try:
        from src.research.hypothesis_validation import AutonomousHypothesisValidator, ValidationMethod
        
        print("   • Initializing Autonomous Hypothesis Validator...")
        validator = AutonomousHypothesisValidator(
            default_alpha=0.05,
            min_effect_size=0.2,
            min_power=0.8
        )
        
        # Test multiple hypotheses
        hypotheses_to_test = [
            ("The mean value significantly differs from zero", validation_data),
            ("The data follows a normal distribution pattern", validation_data),
            ("There is a significant trend in the measurements", validation_data)
        ]
        
        print("   • Designing and executing validation experiments...")
        validation_results = validator.batch_validate_hypotheses(
            hypotheses_to_test,
            domain="statistics"
        )
        
        print(f"   ✅ Validated {len(validation_results)} hypotheses:")
        for i, result in enumerate(validation_results, 1):
            status = "SUPPORTED" if result.hypothesis_supported else "NOT SUPPORTED"
            print(f"      {i}. {status} (p={result.p_value:.4f}, "
                  f"effect={result.effect_size:.3f}, quality={result.get_quality_score():.3f})")
        
        # Generate comprehensive validation report
        print("   • Generating validation report...")
        report = validator.generate_validation_report(validation_results)
        
        print("   📊 Validation summary:")
        summary = report['summary']
        print(f"      • Support rate: {summary['support_rate']:.1%}")
        print(f"      • Average p-value: {summary['average_p_value']:.4f}")
        print(f"      • Average effect size: {summary['average_effect_size']:.3f}")
        print(f"      • Average quality: {summary['average_quality_score']:.3f}")
        
        print("   💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"      • {rec}")
        
    except Exception as e:
        print(f"   ❌ Hypothesis validation failed: {e}")


def demonstrate_integrated_pipeline(data_package):
    """Demonstrate integrated scientific discovery pipeline"""
    
    print("   • Assembling integrated scientific discovery pipeline...")
    
    try:
        # Import all components
        from src.algorithms.causal_discovery import CausalDiscoveryEngine
        from src.research.multimodal_reasoning import MultiModalReasoningEngine
        from src.research.hypothesis_validation import AutonomousHypothesisValidator
        from src.research.autonomous_researcher import AutonomousResearcher
        from src.algorithms.breakthrough_ml import AdaptiveMetaLearner
        
        print("   • Step 1: Autonomous hypothesis generation...")
        researcher = AutonomousResearcher(research_domain="thermodynamics")
        
        # Generate research hypotheses from data
        primary_hypothesis = researcher.generate_research_hypothesis(
            data=data_package['causal_data'][:, 0],  # Temperature data
            context="Temperature effects in thermodynamic systems",
            prior_knowledge={'domain': 'thermodynamics'}
        )
        
        print(f"      ✓ Generated hypothesis: {primary_hypothesis.title}")
        
        print("   • Step 2: Multi-modal evidence integration...")
        reasoning_engine = MultiModalReasoningEngine()
        
        reasoning_result = reasoning_engine.holistic_scientific_reasoning(
            text_data=data_package['text_data'],
            numerical_data=data_package['numerical_data'],
            domain="thermodynamics"
        )
        
        print(f"      ✓ Integrated {len(reasoning_result.hypotheses)} hypotheses from multi-modal evidence")
        
        print("   • Step 3: Causal relationship discovery...")
        causal_engine = CausalDiscoveryEngine(min_causal_strength=0.25)
        
        causal_graph = causal_engine.discover_causal_structure(
            data=data_package['causal_data'][:, :3],  # Temperature, Pressure, Density
            variable_names=['Temperature', 'Pressure', 'Density']
        )
        
        print(f"      ✓ Discovered {len(causal_graph.edges)} causal relationships")
        
        print("   • Step 4: Breakthrough algorithm analysis...")
        meta_learner = AdaptiveMetaLearner(adaptation_rate=0.02)
        
        breakthrough_result = meta_learner.execute(
            data=data_package['causal_data'][:100],  # Sample for speed
            targets=None
        )
        
        print(f"      ✓ Breakthrough score: {breakthrough_result.breakthrough_score:.3f}")
        print(f"      ✓ Novel insights: {len(breakthrough_result.novel_insights)}")
        
        print("   • Step 5: Autonomous validation...")
        validator = AutonomousHypothesisValidator()
        
        # Create hypotheses from reasoning results
        test_hypotheses = [
            (hyp.claim, data_package['validation_data']) 
            for hyp in reasoning_result.hypotheses[:2]
        ]
        
        validation_results = validator.batch_validate_hypotheses(test_hypotheses)
        supported_count = sum(1 for r in validation_results if r.hypothesis_supported)
        
        print(f"      ✓ Validated {supported_count}/{len(validation_results)} hypotheses")
        
        print("   • Step 6: Integrated insights synthesis...")
        
        # Synthesize all results
        integrated_insights = {
            'total_hypotheses_generated': len(reasoning_result.hypotheses) + 1,
            'causal_relationships_discovered': len(causal_graph.edges),
            'breakthrough_algorithms_score': breakthrough_result.breakthrough_score,
            'validation_success_rate': supported_count / len(validation_results) if validation_results else 0,
            'cross_modal_insights': len(reasoning_result.cross_modal_insights),
            'novel_connections': len(reasoning_result.novel_connections),
            'overall_confidence': np.mean([
                reasoning_result.quality_metrics.get('novelty_confidence_balance', 0.5),
                breakthrough_result.breakthrough_score,
                supported_count / max(1, len(validation_results))
            ])
        }
        
        print("   🏆 INTEGRATED PIPELINE RESULTS:")
        print(f"      • Hypotheses generated: {integrated_insights['total_hypotheses_generated']}")
        print(f"      • Causal relationships: {integrated_insights['causal_relationships_discovered']}")
        print(f"      • Breakthrough score: {integrated_insights['breakthrough_algorithms_score']:.3f}")
        print(f"      • Validation success: {integrated_insights['validation_success_rate']:.1%}")
        print(f"      • Cross-modal insights: {integrated_insights['cross_modal_insights']}")
        print(f"      • Overall confidence: {integrated_insights['overall_confidence']:.3f}")
        
        # Export comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_generation1_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            # Convert numpy types to JSON-serializable types
            serializable_insights = {}
            for key, value in integrated_insights.items():
                if isinstance(value, np.number):
                    serializable_insights[key] = float(value)
                else:
                    serializable_insights[key] = value
            
            json.dump({
                'timestamp': timestamp,
                'integrated_insights': serializable_insights,
                'causal_graph_summary': {
                    'nodes': list(causal_graph.nodes),
                    'edge_count': len(causal_graph.edges),
                    'discovery_method': causal_graph.discovery_method
                },
                'reasoning_summary': {
                    'hypothesis_count': len(reasoning_result.hypotheses),
                    'insight_count': len(reasoning_result.cross_modal_insights),
                    'quality_metrics': reasoning_result.quality_metrics
                }
            }, f, indent=2)
        
        print(f"   💾 Complete results exported to: {results_file}")
        
    except Exception as e:
        print(f"   ❌ Integrated pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()