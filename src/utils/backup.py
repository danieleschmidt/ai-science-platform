"""Backup and recovery utilities for AI Science Platform"""

import os
import shutil
import json
import time
import gzip
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for backup operations"""
    backup_id: str
    timestamp: str
    source_path: str
    backup_path: str
    file_count: int
    total_size_bytes: int
    checksum: str
    compression: bool
    backup_type: str  # 'full', 'incremental', 'snapshot'
    status: str  # 'in_progress', 'completed', 'failed'
    error_message: Optional[str] = None


class BackupManager:
    """Manages backup and recovery operations"""
    
    def __init__(self, backup_root: str = "backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.backup_root / "backup_metadata.json"
        self.backups: Dict[str, BackupMetadata] = {}
        
        # Load existing backup metadata
        self._load_metadata()
        
        logger.info(f"BackupManager initialized with root: {self.backup_root}")
    
    def _load_metadata(self) -> None:
        """Load backup metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.backups = {
                        backup_id: BackupMetadata(**backup_data)
                        for backup_id, backup_data in data.items()
                    }
                logger.info(f"Loaded metadata for {len(self.backups)} backups")
            except Exception as e:
                logger.error(f"Failed to load backup metadata: {e}")
                self.backups = {}
    
    def _save_metadata(self) -> None:
        """Save backup metadata to file"""
        try:
            data = {
                backup_id: asdict(metadata)
                for backup_id, metadata in self.backups.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")
    
    def create_backup(self, 
                     source_path: str,
                     backup_name: Optional[str] = None,
                     backup_type: str = "full",
                     compression: bool = True) -> str:
        """Create a backup of the specified path"""
        
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
        
        # Generate backup ID and paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = backup_name or f"backup_{timestamp}"
        backup_dir = self.backup_root / backup_id
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.now().isoformat(),
            source_path=str(source.absolute()),
            backup_path=str(backup_dir),
            file_count=0,
            total_size_bytes=0,
            checksum="",
            compression=compression,
            backup_type=backup_type,
            status="in_progress"
        )
        
        self.backups[backup_id] = metadata
        self._save_metadata()
        
        try:
            logger.info(f"Starting backup: {backup_id} from {source_path}")
            
            if source.is_file():
                file_count, total_size, checksum = self._backup_file(source, backup_dir, compression)
            else:
                file_count, total_size, checksum = self._backup_directory(source, backup_dir, compression)
            
            # Update metadata
            metadata.file_count = file_count
            metadata.total_size_bytes = total_size
            metadata.checksum = checksum
            metadata.status = "completed"
            
            self.backups[backup_id] = metadata
            self._save_metadata()
            
            logger.info(f"Backup completed: {backup_id} ({file_count} files, {total_size / 1024:.1f} KB)")
            return backup_id
            
        except Exception as e:
            metadata.status = "failed"
            metadata.error_message = str(e)
            self.backups[backup_id] = metadata
            self._save_metadata()
            logger.error(f"Backup failed: {backup_id} - {e}")
            raise
    
    def _backup_file(self, source: Path, backup_dir: Path, compression: bool) -> tuple:
        """Backup a single file"""
        dest_path = backup_dir / source.name
        
        if compression and source.suffix not in ['.gz', '.zip', '.tar']:
            dest_path = dest_path.with_suffix(dest_path.suffix + '.gz')
            with open(source, 'rb') as f_in:
                with gzip.open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(source, dest_path)
        
        file_size = dest_path.stat().st_size
        checksum = self._calculate_checksum(dest_path)
        
        return 1, file_size, checksum
    
    def _backup_directory(self, source: Path, backup_dir: Path, compression: bool) -> tuple:
        """Backup a directory recursively"""
        total_files = 0
        total_size = 0
        checksums = []
        
        for item in source.rglob('*'):
            if item.is_file():
                # Calculate relative path
                rel_path = item.relative_to(source)
                dest_path = backup_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file with optional compression
                if compression and item.suffix not in ['.gz', '.zip', '.tar'] and item.stat().st_size > 1024:
                    dest_path = dest_path.with_suffix(dest_path.suffix + '.gz')
                    with open(item, 'rb') as f_in:
                        with gzip.open(dest_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy2(item, dest_path)
                
                total_files += 1
                file_size = dest_path.stat().st_size
                total_size += file_size
                checksums.append(self._calculate_checksum(dest_path))
        
        # Calculate overall checksum
        overall_checksum = hashlib.sha256(''.join(sorted(checksums)).encode()).hexdigest()
        
        return total_files, total_size, overall_checksum
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        else:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def restore_backup(self, backup_id: str, restore_path: str, overwrite: bool = False) -> bool:
        """Restore a backup to the specified path"""
        if backup_id not in self.backups:
            raise ValueError(f"Backup not found: {backup_id}")
        
        metadata = self.backups[backup_id]
        if metadata.status != "completed":
            raise ValueError(f"Cannot restore incomplete backup: {backup_id}")
        
        backup_dir = Path(metadata.backup_path)
        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup directory not found: {backup_dir}")
        
        restore_dest = Path(restore_path)
        
        if restore_dest.exists() and not overwrite:
            raise FileExistsError(f"Restore destination exists and overwrite=False: {restore_path}")
        
        try:
            logger.info(f"Starting restore: {backup_id} to {restore_path}")
            
            # Remove destination if overwriting
            if restore_dest.exists():
                if restore_dest.is_file():
                    restore_dest.unlink()
                else:
                    shutil.rmtree(restore_dest)
            
            # Restore files
            files_restored = 0
            for item in backup_dir.rglob('*'):
                if item.is_file():
                    # Calculate destination path
                    rel_path = item.relative_to(backup_dir)
                    dest_file = restore_dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Restore file with decompression if needed
                    if item.suffix == '.gz' and not rel_path.suffix == '.gz':
                        # Decompress file
                        with gzip.open(item, 'rb') as f_in:
                            with open(dest_file.with_suffix(''), 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    else:
                        shutil.copy2(item, dest_file)
                    
                    files_restored += 1
            
            logger.info(f"Restore completed: {backup_id} ({files_restored} files restored)")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {backup_id} - {e}")
            raise
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups"""
        return list(self.backups.values())
    
    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get information about a specific backup"""
        return self.backups.get(backup_id)
    
    def verify_backup(self, backup_id: str) -> bool:
        """Verify the integrity of a backup"""
        if backup_id not in self.backups:
            raise ValueError(f"Backup not found: {backup_id}")
        
        metadata = self.backups[backup_id]
        backup_dir = Path(metadata.backup_path)
        
        if not backup_dir.exists():
            logger.error(f"Backup directory not found: {backup_dir}")
            return False
        
        try:
            # Count files and calculate checksums
            file_count = 0
            checksums = []
            
            for item in backup_dir.rglob('*'):
                if item.is_file():
                    file_count += 1
                    checksums.append(self._calculate_checksum(item))
            
            # Verify file count
            if file_count != metadata.file_count:
                logger.error(f"File count mismatch: expected {metadata.file_count}, found {file_count}")
                return False
            
            # Verify overall checksum
            overall_checksum = hashlib.sha256(''.join(sorted(checksums)).encode()).hexdigest()
            if overall_checksum != metadata.checksum:
                logger.error(f"Checksum mismatch: expected {metadata.checksum}, calculated {overall_checksum}")
                return False
            
            logger.info(f"Backup verification successful: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {backup_id} - {e}")
            return False
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        if backup_id not in self.backups:
            raise ValueError(f"Backup not found: {backup_id}")
        
        metadata = self.backups[backup_id]
        backup_dir = Path(metadata.backup_path)
        
        try:
            # Remove backup directory
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            
            # Remove from metadata
            del self.backups[backup_id]
            self._save_metadata()
            
            logger.info(f"Backup deleted: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def cleanup_old_backups(self, max_age_days: int = 30, max_count: int = 10) -> int:
        """Clean up old backups based on age and count"""
        deleted_count = 0
        
        # Get backups sorted by timestamp (oldest first)
        sorted_backups = sorted(
            self.backups.items(),
            key=lambda x: x[1].timestamp
        )
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        # Delete old backups
        for backup_id, metadata in sorted_backups:
            backup_time = datetime.fromisoformat(metadata.timestamp).timestamp()
            
            # Check if backup is too old or we have too many
            if (backup_time < cutoff_time or 
                len(self.backups) - deleted_count > max_count):
                
                try:
                    if self.delete_backup(backup_id):
                        deleted_count += 1
                        logger.info(f"Cleaned up old backup: {backup_id}")
                except Exception as e:
                    logger.error(f"Failed to clean up backup {backup_id}: {e}")
        
        return deleted_count


# Global backup manager instance
_backup_manager = None


def get_backup_manager(backup_root: str = "backups") -> BackupManager:
    """Get or create global backup manager"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager(backup_root)
    return _backup_manager


def backup_experiments(backup_name: Optional[str] = None) -> str:
    """Backup experiment results directory"""
    manager = get_backup_manager()
    return manager.create_backup("experiment_results", backup_name)


def backup_logs(backup_name: Optional[str] = None) -> str:
    """Backup logs directory"""
    manager = get_backup_manager()
    return manager.create_backup("logs", backup_name)


def backup_models(models_path: str = "models", backup_name: Optional[str] = None) -> str:
    """Backup saved models"""
    manager = get_backup_manager()
    if Path(models_path).exists():
        return manager.create_backup(models_path, backup_name)
    else:
        raise FileNotFoundError(f"Models directory not found: {models_path}")