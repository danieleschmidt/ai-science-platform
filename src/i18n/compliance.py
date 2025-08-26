"""Global compliance management (GDPR, CCPA, PDPA)"""

import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
import threading
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceRegion(Enum):
    """Supported compliance regions"""
    EU = "eu"          # GDPR
    CALIFORNIA = "ca"  # CCPA
    SINGAPORE = "sg"   # PDPA
    BRAZIL = "br"      # LGPD
    CANADA = "can"     # PIPEDA
    UK = "uk"          # UK GDPR
    AUSTRALIA = "au"   # Privacy Act


@dataclass
class DataProcessingRecord:
    """Record of data processing activity"""
    id: str
    timestamp: float
    data_type: str
    processing_purpose: str
    data_subject: str
    lawful_basis: str
    retention_period: int  # days
    region: ComplianceRegion
    consent_given: bool = False
    pseudonymized: bool = False
    encrypted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'data_type': self.data_type,
            'processing_purpose': self.processing_purpose,
            'data_subject': self.data_subject,
            'lawful_basis': self.lawful_basis,
            'retention_period': self.retention_period,
            'region': self.region.value,
            'consent_given': self.consent_given,
            'pseudonymized': self.pseudonymized,
            'encrypted': self.encrypted
        }


@dataclass
class ConsentRecord:
    """User consent record"""
    subject_id: str
    consent_type: str
    granted: bool
    timestamp: float
    expires_at: Optional[float] = None
    withdrawn_at: Optional[float] = None
    purpose: str = ""
    region: ComplianceRegion = ComplianceRegion.EU
    
    def is_valid(self) -> bool:
        """Check if consent is still valid"""
        now = time.time()
        
        if self.withdrawn_at and now > self.withdrawn_at:
            return False
        
        if self.expires_at and now > self.expires_at:
            return False
        
        return self.granted


class ComplianceManager:
    """Global compliance management system"""
    
    def __init__(self):
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.active_regions: Set[ComplianceRegion] = set()
        self.lock = threading.RLock()
        
        # Compliance settings
        self.settings = {
            'data_retention_days': {
                ComplianceRegion.EU: 1095,      # 3 years default
                ComplianceRegion.CALIFORNIA: 365, # 1 year default
                ComplianceRegion.SINGAPORE: 1095,
                ComplianceRegion.BRAZIL: 1095,
                ComplianceRegion.CANADA: 1095,
                ComplianceRegion.UK: 1095,
                ComplianceRegion.AUSTRALIA: 1095
            },
            'require_consent': {
                ComplianceRegion.EU: True,
                ComplianceRegion.CALIFORNIA: False,
                ComplianceRegion.SINGAPORE: True,
                ComplianceRegion.BRAZIL: True,
                ComplianceRegion.CANADA: True,
                ComplianceRegion.UK: True,
                ComplianceRegion.AUSTRALIA: False
            },
            'right_to_deletion': True,
            'right_to_portability': True,
            'breach_notification_hours': 72,
            'pseudonymization_required': True,
            'encryption_at_rest': True
        }
        
        logger.info("ComplianceManager initialized")
    
    def enable_region(self, region: ComplianceRegion) -> None:
        """Enable compliance for a specific region"""
        with self.lock:
            self.active_regions.add(region)
            logger.info(f"Enabled compliance for region: {region.value}")
    
    def disable_region(self, region: ComplianceRegion) -> None:
        """Disable compliance for a specific region"""
        with self.lock:
            self.active_regions.discard(region)
            logger.info(f"Disabled compliance for region: {region.value}")
    
    def record_processing(self, data_type: str, purpose: str, subject_id: str,
                         region: ComplianceRegion, lawful_basis: str = "legitimate_interest") -> str:
        """Record data processing activity"""
        with self.lock:
            record_id = hashlib.md5(f"{subject_id}_{data_type}_{purpose}_{time.time()}".encode()).hexdigest()
            
            record = DataProcessingRecord(
                id=record_id,
                timestamp=time.time(),
                data_type=data_type,
                processing_purpose=purpose,
                data_subject=subject_id,
                lawful_basis=lawful_basis,
                retention_period=self.settings['data_retention_days'][region],
                region=region,
                pseudonymized=self.settings['pseudonymization_required'],
                encrypted=self.settings['encryption_at_rest']
            )
            
            self.processing_records.append(record)
            logger.debug(f"Recorded processing activity: {record_id}")
            
            return record_id
    
    def request_consent(self, subject_id: str, consent_type: str, purpose: str,
                       region: ComplianceRegion, expires_days: Optional[int] = None) -> bool:
        """Request and record consent"""
        with self.lock:
            # Check if consent is required for this region
            if not self.settings['require_consent'].get(region, False):
                return True  # Consent not required
            
            expires_at = None
            if expires_days:
                expires_at = time.time() + (expires_days * 24 * 3600)
            
            consent = ConsentRecord(
                subject_id=subject_id,
                consent_type=consent_type,
                granted=True,  # In real implementation, would prompt user
                timestamp=time.time(),
                expires_at=expires_at,
                purpose=purpose,
                region=region
            )
            
            self.consent_records[f"{subject_id}_{consent_type}"] = consent
            logger.info(f"Consent granted for {subject_id}: {consent_type}")
            
            return True
    
    def withdraw_consent(self, subject_id: str, consent_type: str) -> bool:
        """Withdraw consent for a subject"""
        with self.lock:
            consent_key = f"{subject_id}_{consent_type}"
            
            if consent_key in self.consent_records:
                self.consent_records[consent_key].withdrawn_at = time.time()
                self.consent_records[consent_key].granted = False
                
                logger.info(f"Consent withdrawn for {subject_id}: {consent_type}")
                
                # Trigger data deletion if required
                self._handle_consent_withdrawal(subject_id, consent_type)
                
                return True
            
            return False
    
    def check_consent(self, subject_id: str, consent_type: str) -> bool:
        """Check if valid consent exists"""
        with self.lock:
            consent_key = f"{subject_id}_{consent_type}"
            
            if consent_key in self.consent_records:
                return self.consent_records[consent_key].is_valid()
            
            return False
    
    def handle_data_subject_request(self, subject_id: str, request_type: str,
                                  region: ComplianceRegion) -> Dict[str, Any]:
        """Handle data subject requests (access, deletion, portability)"""
        with self.lock:
            response = {
                'subject_id': subject_id,
                'request_type': request_type,
                'region': region.value,
                'timestamp': time.time(),
                'status': 'processed',
                'data': {}
            }
            
            if request_type == 'access':
                # Right to access
                subject_records = [r for r in self.processing_records if r.data_subject == subject_id]
                response['data'] = {
                    'processing_records': [r.to_dict() for r in subject_records],
                    'consent_records': {k: v.__dict__ for k, v in self.consent_records.items() 
                                      if v.subject_id == subject_id}
                }
                
            elif request_type == 'deletion':
                # Right to be forgotten
                if self.settings['right_to_deletion']:
                    self._delete_subject_data(subject_id)
                    response['data']['deleted'] = True
                else:
                    response['status'] = 'denied'
                    response['reason'] = 'Deletion not supported in this region'
                    
            elif request_type == 'portability':
                # Right to data portability
                if self.settings['right_to_portability']:
                    subject_data = self._export_subject_data(subject_id)
                    response['data'] = subject_data
                else:
                    response['status'] = 'denied'
                    response['reason'] = 'Data portability not supported'
            
            logger.info(f"Processed data subject request: {request_type} for {subject_id}")
            return response
    
    def _handle_consent_withdrawal(self, subject_id: str, consent_type: str):
        """Handle consent withdrawal by deleting associated data"""
        # Remove processing records where consent was the lawful basis
        records_to_remove = []
        for record in self.processing_records:
            if (record.data_subject == subject_id and 
                record.lawful_basis == 'consent'):
                records_to_remove.append(record)
        
        for record in records_to_remove:
            self.processing_records.remove(record)
        
        logger.info(f"Removed {len(records_to_remove)} records due to consent withdrawal")
    
    def _delete_subject_data(self, subject_id: str):
        """Delete all data for a subject"""
        # Remove processing records
        initial_count = len(self.processing_records)
        self.processing_records = [r for r in self.processing_records if r.data_subject != subject_id]
        removed_count = initial_count - len(self.processing_records)
        
        # Remove consent records
        consent_keys_to_remove = [k for k, v in self.consent_records.items() 
                                if v.subject_id == subject_id]
        for key in consent_keys_to_remove:
            del self.consent_records[key]
        
        logger.info(f"Deleted data for subject {subject_id}: "
                   f"{removed_count} processing records, {len(consent_keys_to_remove)} consent records")
    
    def _export_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Export all data for a subject in portable format"""
        subject_records = [r for r in self.processing_records if r.data_subject == subject_id]
        subject_consents = {k: v.__dict__ for k, v in self.consent_records.items() 
                          if v.subject_id == subject_id}
        
        return {
            'subject_id': subject_id,
            'export_timestamp': time.time(),
            'processing_activities': [r.to_dict() for r in subject_records],
            'consent_history': subject_consents
        }
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired data based on retention policies"""
        with self.lock:
            now = time.time()
            initial_count = len(self.processing_records)
            
            # Remove expired processing records
            self.processing_records = [
                r for r in self.processing_records
                if now - r.timestamp < (r.retention_period * 24 * 3600)
            ]
            
            # Remove expired consent records
            expired_consents = [
                k for k, v in self.consent_records.items()
                if v.expires_at and now > v.expires_at
            ]
            
            for key in expired_consents:
                del self.consent_records[key]
            
            removed_records = initial_count - len(self.processing_records)
            removed_consents = len(expired_consents)
            
            if removed_records > 0 or removed_consents > 0:
                logger.info(f"Cleaned up expired data: "
                           f"{removed_records} processing records, {removed_consents} consent records")
            
            return removed_records + removed_consents
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        with self.lock:
            now = time.time()
            
            # Calculate consent statistics
            total_consents = len(self.consent_records)
            active_consents = sum(1 for c in self.consent_records.values() if c.is_valid())
            
            # Calculate processing statistics by region
            region_stats = {}
            for region in self.active_regions:
                region_records = [r for r in self.processing_records if r.region == region]
                region_stats[region.value] = {
                    'total_records': len(region_records),
                    'encrypted_records': sum(1 for r in region_records if r.encrypted),
                    'pseudonymized_records': sum(1 for r in region_records if r.pseudonymized)
                }
            
            return {
                'timestamp': now,
                'active_regions': [r.value for r in self.active_regions],
                'total_processing_records': len(self.processing_records),
                'total_consent_records': total_consents,
                'active_consent_records': active_consents,
                'consent_compliance_rate': active_consents / max(1, total_consents),
                'region_statistics': region_stats,
                'data_protection_measures': {
                    'encryption_enabled': self.settings['encryption_at_rest'],
                    'pseudonymization_enabled': self.settings['pseudonymization_required'],
                    'right_to_deletion_enabled': self.settings['right_to_deletion'],
                    'right_to_portability_enabled': self.settings['right_to_portability']
                },
                'settings': self.settings
            }
    
    def export_audit_log(self, filepath: str):
        """Export audit log for compliance reporting"""
        report = self.get_compliance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Compliance audit log exported to {filepath}")


# Global compliance manager instance
_compliance_manager = None

def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager instance"""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager()
        # Enable key regions by default
        _compliance_manager.enable_region(ComplianceRegion.EU)
        _compliance_manager.enable_region(ComplianceRegion.CALIFORNIA)
    return _compliance_manager


# Example usage
if __name__ == "__main__":
    manager = ComplianceManager()
    
    # Enable regions
    manager.enable_region(ComplianceRegion.EU)
    manager.enable_region(ComplianceRegion.CALIFORNIA)
    
    # Request consent
    manager.request_consent("user123", "analytics", "scientific research", ComplianceRegion.EU)
    
    # Record processing
    record_id = manager.record_processing(
        "research_data", 
        "scientific_discovery", 
        "user123", 
        ComplianceRegion.EU,
        "consent"
    )
    
    # Generate report
    report = manager.get_compliance_report()
    print("Compliance Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # Test data subject request
    access_response = manager.handle_data_subject_request("user123", "access", ComplianceRegion.EU)
    print(f"\nData Subject Access Response:")
    print(f"  Status: {access_response['status']}")
    print(f"  Records: {len(access_response['data']['processing_records'])}")