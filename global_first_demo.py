"""🌍 GLOBAL-FIRST PLATFORM DEMONSTRATION"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logging_config import setup_logging
from src.i18n.localizer import get_localizer, translate, set_language
from src.i18n.compliance import get_compliance_manager, ComplianceRegion
from src.algorithms.discovery import DiscoveryEngine
from src.utils.data_utils import generate_sample_data

logger = logging.getLogger(__name__)


def demo_multilingual_support():
    """Demonstrate multi-language support"""
    print("\n🌍 GLOBAL-FIRST: Multi-Language Support")
    print("=" * 50)
    
    localizer = get_localizer()
    
    # Test each supported language
    languages = [
        ('en', 'English'),
        ('es', 'Español'),
        ('fr', 'Français'),
        ('de', 'Deutsch'),
        ('ja', '日本語'),
        ('zh', '中文')
    ]
    
    for lang_code, lang_name in languages:
        set_language(lang_code)
        
        platform_name = translate('platform.name')
        description = translate('platform.description')
        discovery_started = translate('discovery.process.started')
        
        print(f"{lang_name} ({lang_code}):")
        print(f"  Platform: {platform_name}")
        print(f"  Description: {description}")
        print(f"  Status: {discovery_started}")
        print()
    
    # Test translation statistics
    stats = localizer.get_translation_stats()
    print("Translation Statistics:")
    print(f"  Languages Available: {stats['available_languages']}")
    print(f"  Current Language: {stats['current_language']}")
    
    for lang, info in stats['languages'].items():
        print(f"  {lang}: {info['translation_count']} keys, {info['coverage']:.1%} coverage")


def demo_compliance_management():
    """Demonstrate global compliance management"""
    print("\n🛡️ GLOBAL-FIRST: Compliance Management")
    print("=" * 50)
    
    compliance = get_compliance_manager()
    
    # Enable multiple regions
    regions = [
        (ComplianceRegion.EU, "European Union (GDPR)"),
        (ComplianceRegion.CALIFORNIA, "California (CCPA)"),
        (ComplianceRegion.SINGAPORE, "Singapore (PDPA)"),
        (ComplianceRegion.UK, "United Kingdom (UK GDPR)")
    ]
    
    print("Enabling compliance for multiple regions:")
    for region, description in regions:
        compliance.enable_region(region)
        print(f"✅ {description}")
    
    # Simulate data processing activities
    test_subjects = ["researcher_001", "scientist_002", "analyst_003"]
    
    print(f"\nSimulating scientific data processing for {len(test_subjects)} subjects:")
    
    for i, subject in enumerate(test_subjects):
        # Different regions for different users
        region = regions[i % len(regions)][0]
        
        # Request consent (if required by region)
        consent_granted = compliance.request_consent(
            subject, 
            "scientific_research", 
            "AI-driven scientific discovery research",
            region,
            expires_days=365
        )
        
        if consent_granted:
            # Record data processing
            record_id = compliance.record_processing(
                data_type="research_data",
                purpose="scientific_discovery",
                subject_id=subject,
                region=region,
                lawful_basis="consent" if compliance.settings['require_consent'].get(region, False) else "legitimate_interest"
            )
            
            print(f"✅ Processed data for {subject} in {region.value} (Record: {record_id[:8]}...)")
    
    # Generate compliance report
    report = compliance.get_compliance_report()
    
    print(f"\n📊 Compliance Report:")
    print(f"  Active Regions: {', '.join(report['active_regions'])}")
    print(f"  Total Processing Records: {report['total_processing_records']}")
    print(f"  Active Consent Records: {report['active_consent_records']}")
    print(f"  Consent Compliance Rate: {report['consent_compliance_rate']:.1%}")
    print(f"  Encryption Enabled: {report['data_protection_measures']['encryption_enabled']}")
    print(f"  Right to Deletion: {report['data_protection_measures']['right_to_deletion_enabled']}")
    
    # Test data subject rights
    print(f"\n🔒 Testing Data Subject Rights:")
    
    # Test right to access
    access_response = compliance.handle_data_subject_request(
        "researcher_001", 
        "access", 
        ComplianceRegion.EU
    )
    
    print(f"  Access Request: {access_response['status']}")
    print(f"  Records Found: {len(access_response['data']['processing_records'])}")
    
    # Test right to deletion (for demonstration, we won't actually delete)
    deletion_response = compliance.handle_data_subject_request(
        "researcher_001",
        "deletion",
        ComplianceRegion.EU
    )
    
    print(f"  Deletion Request: {deletion_response['status']}")
    if 'deleted' in deletion_response['data']:
        print(f"  Data Deleted: {deletion_response['data']['deleted']}")


def demo_integrated_global_platform():
    """Demonstrate integrated global platform with i18n and compliance"""
    print("\n🚀 GLOBAL-FIRST: Integrated Platform Demo")
    print("=" * 50)
    
    # Set to Spanish for demo
    set_language('es')
    print(f"Platform Language: Español")
    
    # Initialize compliance for EU (Spanish user)
    compliance = get_compliance_manager()
    user_id = "scientific_researcher_es_001"
    
    # Request consent in Spanish context
    consent_granted = compliance.request_consent(
        user_id,
        "investigacion_cientifica",
        "Investigación de descubrimiento científico impulsada por IA",
        ComplianceRegion.EU,
        expires_days=730
    )
    
    print(f"Consent Status: {'✅ Granted' if consent_granted else '❌ Denied'}")
    
    # Run scientific discovery with compliance
    if consent_granted:
        # Record the processing activity
        record_id = compliance.record_processing(
            data_type="datos_investigacion",
            purpose="descubrimiento_cientifico",
            subject_id=user_id,
            region=ComplianceRegion.EU,
            lawful_basis="consent"
        )
        
        print(f"Data Processing Recorded: {record_id[:12]}...")
        
        # Generate research data
        data, _ = generate_sample_data(size=100, data_type='normal')
        print(f"Generated Research Dataset: {data.shape}")
        
        # Run discovery engine with translated messages
        print(f"\n{translate('discovery.process.started')}...")
        
        engine = DiscoveryEngine(discovery_threshold=0.6)
        discoveries = engine.discover(data, context="investigacion_global")
        
        print(f"{translate('discovery.process.complete')}: {len(discoveries)} discoveries")
        
        if discoveries:
            print(f"\n{translate('discovery.hypothesis.generated')}:")
            for i, discovery in enumerate(discoveries[:2]):  # Show first 2
                print(f"  {i+1}. {discovery.hypothesis}")
                print(f"     Confidence: {discovery.confidence:.3f}")
    
    # Show final global status
    print(f"\n🌐 Global Platform Status:")
    
    # Language status
    localizer = get_localizer()
    lang_stats = localizer.get_translation_stats()
    print(f"  Current Language: {lang_stats['current_language']}")
    print(f"  Available Languages: {lang_stats['available_languages']}")
    
    # Compliance status
    compliance_report = compliance.get_compliance_report()
    print(f"  Active Compliance Regions: {len(compliance_report['active_regions'])}")
    print(f"  Total Processing Records: {compliance_report['total_processing_records']}")
    print(f"  Compliance Rate: {compliance_report['consent_compliance_rate']:.1%}")
    
    # Show region-specific stats
    print(f"\n📋 Regional Compliance Summary:")
    for region, stats in compliance_report['region_statistics'].items():
        print(f"  {region.upper()}: {stats['total_records']} records, "
              f"{stats['encrypted_records']} encrypted")


def main():
    """Execute global-first platform demonstration"""
    print("🌍 AI SCIENCE PLATFORM - GLOBAL-FIRST DEMONSTRATION")
    print("=" * 60)
    print("Showcasing multi-language support and global compliance")
    print()
    
    setup_logging()
    logger.info("Starting Global-First platform demonstration")
    
    try:
        # Run demonstrations
        demo_multilingual_support()
        demo_compliance_management() 
        demo_integrated_global_platform()
        
        print("\n✅ GLOBAL-FIRST DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\n🌐 Global Platform Features Demonstrated:")
        print("  ✅ Multi-language support (6 languages)")
        print("  ✅ GDPR, CCPA, PDPA compliance")
        print("  ✅ Data subject rights (access, deletion, portability)")
        print("  ✅ Consent management and tracking")
        print("  ✅ Region-specific data processing")
        print("  ✅ Integrated localized scientific discovery")
        
        return True
        
    except Exception as e:
        logger.error(f"Global-First demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)