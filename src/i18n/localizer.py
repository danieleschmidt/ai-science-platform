"""Multi-language localization system"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class Localizer:
    """Multi-language localization manager"""
    
    def __init__(self, default_language: str = 'en'):
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self.lock = threading.RLock()
        
        # Load built-in translations
        self._load_builtin_translations()
        
        logger.info(f"Localizer initialized with default language: {default_language}")
    
    def _load_builtin_translations(self):
        """Load built-in translations for supported languages"""
        
        # English (default)
        self.translations['en'] = {
            'platform.name': 'AI Science Platform',
            'platform.description': 'AI-driven scientific discovery automation',
            'discovery.engine.initialized': 'Discovery engine initialized',
            'discovery.process.started': 'Discovery process started',
            'discovery.process.complete': 'Discovery process complete',
            'discovery.hypothesis.generated': 'Generated hypothesis',
            'discovery.hypothesis.test.passed': 'Hypothesis test passed',
            'discovery.hypothesis.test.failed': 'Hypothesis test failed',
            'model.training.started': 'Model training started',
            'model.training.complete': 'Training completed',
            'model.prediction.made': 'Prediction made',
            'error.invalid.data': 'Invalid data provided',
            'error.model.not.trained': 'Model not trained',
            'error.processing.failed': 'Processing failed',
            'cache.hit': 'Cache hit',
            'cache.miss': 'Cache miss',
            'performance.optimization.enabled': 'Performance optimization enabled',
            'scaling.up': 'Scaling up workers',
            'scaling.down': 'Scaling down workers',
            'quality.gate.passed': 'Quality gate passed',
            'quality.gate.failed': 'Quality gate failed',
            'security.validation.passed': 'Security validation passed',
            'compliance.gdpr.enabled': 'GDPR compliance enabled',
            'compliance.ccpa.enabled': 'CCPA compliance enabled'
        }
        
        # Spanish
        self.translations['es'] = {
            'platform.name': 'Plataforma de Ciencia IA',
            'platform.description': 'Automatización de descubrimientos científicos impulsada por IA',
            'discovery.engine.initialized': 'Motor de descubrimiento inicializado',
            'discovery.process.started': 'Proceso de descubrimiento iniciado',
            'discovery.process.complete': 'Proceso de descubrimiento completo',
            'discovery.hypothesis.generated': 'Hipótesis generada',
            'discovery.hypothesis.test.passed': 'Prueba de hipótesis aprobada',
            'discovery.hypothesis.test.failed': 'Prueba de hipótesis fallida',
            'model.training.started': 'Entrenamiento del modelo iniciado',
            'model.training.complete': 'Entrenamiento completado',
            'model.prediction.made': 'Predicción realizada',
            'error.invalid.data': 'Datos inválidos proporcionados',
            'error.model.not.trained': 'Modelo no entrenado',
            'error.processing.failed': 'Procesamiento fallido',
            'cache.hit': 'Acierto de caché',
            'cache.miss': 'Fallo de caché',
            'performance.optimization.enabled': 'Optimización de rendimiento habilitada',
            'scaling.up': 'Escalando trabajadores hacia arriba',
            'scaling.down': 'Escalando trabajadores hacia abajo',
            'quality.gate.passed': 'Puerta de calidad aprobada',
            'quality.gate.failed': 'Puerta de calidad fallida',
            'security.validation.passed': 'Validación de seguridad aprobada',
            'compliance.gdpr.enabled': 'Cumplimiento GDPR habilitado',
            'compliance.ccpa.enabled': 'Cumplimiento CCPA habilitado'
        }
        
        # French
        self.translations['fr'] = {
            'platform.name': 'Plateforme de Science IA',
            'platform.description': 'Automatisation de la découverte scientifique pilotée par IA',
            'discovery.engine.initialized': 'Moteur de découverte initialisé',
            'discovery.process.started': 'Processus de découverte démarré',
            'discovery.process.complete': 'Processus de découverte terminé',
            'discovery.hypothesis.generated': 'Hypothèse générée',
            'discovery.hypothesis.test.passed': 'Test d\'hypothèse réussi',
            'discovery.hypothesis.test.failed': 'Test d\'hypothèse échoué',
            'model.training.started': 'Entraînement du modèle démarré',
            'model.training.complete': 'Entraînement terminé',
            'model.prediction.made': 'Prédiction effectuée',
            'error.invalid.data': 'Données invalides fournies',
            'error.model.not.trained': 'Modèle non entraîné',
            'error.processing.failed': 'Traitement échoué',
            'cache.hit': 'Succès de cache',
            'cache.miss': 'Échec de cache',
            'performance.optimization.enabled': 'Optimisation des performances activée',
            'scaling.up': 'Augmentation des travailleurs',
            'scaling.down': 'Réduction des travailleurs',
            'quality.gate.passed': 'Porte de qualité réussie',
            'quality.gate.failed': 'Porte de qualité échouée',
            'security.validation.passed': 'Validation de sécurité réussie',
            'compliance.gdpr.enabled': 'Conformité RGPD activée',
            'compliance.ccpa.enabled': 'Conformité CCPA activée'
        }
        
        # German
        self.translations['de'] = {
            'platform.name': 'KI-Wissenschaftsplattform',
            'platform.description': 'KI-gesteuerte wissenschaftliche Entdeckungsautomatisierung',
            'discovery.engine.initialized': 'Entdeckungsmaschine initialisiert',
            'discovery.process.started': 'Entdeckungsprozess gestartet',
            'discovery.process.complete': 'Entdeckungsprozess abgeschlossen',
            'discovery.hypothesis.generated': 'Hypothese generiert',
            'discovery.hypothesis.test.passed': 'Hypothesentest bestanden',
            'discovery.hypothesis.test.failed': 'Hypothesentest fehlgeschlagen',
            'model.training.started': 'Modelltraining gestartet',
            'model.training.complete': 'Training abgeschlossen',
            'model.prediction.made': 'Vorhersage getroffen',
            'error.invalid.data': 'Ungültige Daten bereitgestellt',
            'error.model.not.trained': 'Modell nicht trainiert',
            'error.processing.failed': 'Verarbeitung fehlgeschlagen',
            'cache.hit': 'Cache-Treffer',
            'cache.miss': 'Cache-Fehlschlag',
            'performance.optimization.enabled': 'Leistungsoptimierung aktiviert',
            'scaling.up': 'Arbeiter hochskalieren',
            'scaling.down': 'Arbeiter herunterskalieren',
            'quality.gate.passed': 'Qualitätstor bestanden',
            'quality.gate.failed': 'Qualitätstor fehlgeschlagen',
            'security.validation.passed': 'Sicherheitsvalidierung bestanden',
            'compliance.gdpr.enabled': 'DSGVO-Konformität aktiviert',
            'compliance.ccpa.enabled': 'CCPA-Konformität aktiviert'
        }
        
        # Japanese
        self.translations['ja'] = {
            'platform.name': 'AI科学プラットフォーム',
            'platform.description': 'AI駆動の科学的発見自動化',
            'discovery.engine.initialized': '発見エンジンが初期化されました',
            'discovery.process.started': '発見プロセスが開始されました',
            'discovery.process.complete': '発見プロセスが完了しました',
            'discovery.hypothesis.generated': '仮説が生成されました',
            'discovery.hypothesis.test.passed': '仮説テストに合格しました',
            'discovery.hypothesis.test.failed': '仮説テストに失敗しました',
            'model.training.started': 'モデルトレーニングが開始されました',
            'model.training.complete': 'トレーニングが完了しました',
            'model.prediction.made': '予測が行われました',
            'error.invalid.data': '無効なデータが提供されました',
            'error.model.not.trained': 'モデルがトレーニングされていません',
            'error.processing.failed': '処理に失敗しました',
            'cache.hit': 'キャッシュヒット',
            'cache.miss': 'キャッシュミス',
            'performance.optimization.enabled': 'パフォーマンス最適化が有効になりました',
            'scaling.up': 'ワーカーをスケールアップしています',
            'scaling.down': 'ワーカーをスケールダウンしています',
            'quality.gate.passed': '品質ゲートに合格しました',
            'quality.gate.failed': '品質ゲートに失敗しました',
            'security.validation.passed': 'セキュリティ検証に合格しました',
            'compliance.gdpr.enabled': 'GDPR準拠が有効になりました',
            'compliance.ccpa.enabled': 'CCPA準拠が有効になりました'
        }
        
        # Chinese (Simplified)
        self.translations['zh'] = {
            'platform.name': 'AI科学平台',
            'platform.description': 'AI驱动的科学发现自动化',
            'discovery.engine.initialized': '发现引擎已初始化',
            'discovery.process.started': '发现过程已开始',
            'discovery.process.complete': '发现过程已完成',
            'discovery.hypothesis.generated': '已生成假设',
            'discovery.hypothesis.test.passed': '假设测试通过',
            'discovery.hypothesis.test.failed': '假设测试失败',
            'model.training.started': '模型训练已开始',
            'model.training.complete': '训练已完成',
            'model.prediction.made': '已进行预测',
            'error.invalid.data': '提供的数据无效',
            'error.model.not.trained': '模型未训练',
            'error.processing.failed': '处理失败',
            'cache.hit': '缓存命中',
            'cache.miss': '缓存未命中',
            'performance.optimization.enabled': '性能优化已启用',
            'scaling.up': '扩展工作者',
            'scaling.down': '缩减工作者',
            'quality.gate.passed': '质量门通过',
            'quality.gate.failed': '质量门失败',
            'security.validation.passed': '安全验证通过',
            'compliance.gdpr.enabled': 'GDPR合规已启用',
            'compliance.ccpa.enabled': 'CCPA合规已启用'
        }
    
    def set_language(self, language: str) -> bool:
        """Set the current language"""
        with self.lock:
            if language in self.translations:
                self.current_language = language
                logger.info(f"Language set to: {language}")
                return True
            else:
                logger.warning(f"Language not supported: {language}")
                return False
    
    def get_available_languages(self) -> list:
        """Get list of available languages"""
        return list(self.translations.keys())
    
    def translate(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """Translate a key to the specified language"""
        with self.lock:
            target_language = language or self.current_language
            
            # Try target language
            if target_language in self.translations:
                if key in self.translations[target_language]:
                    text = self.translations[target_language][key]
                    
                    # Apply formatting if kwargs provided
                    if kwargs:
                        try:
                            text = text.format(**kwargs)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Translation formatting error: {e}")
                    
                    return text
            
            # Fallback to default language
            if self.default_language in self.translations:
                if key in self.translations[self.default_language]:
                    text = self.translations[self.default_language][key]
                    
                    if kwargs:
                        try:
                            text = text.format(**kwargs)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Translation formatting error: {e}")
                    
                    return text
            
            # Return key if no translation found
            logger.debug(f"Translation not found: {key}")
            return key
    
    def add_translation(self, language: str, translations: Dict[str, str]):
        """Add or update translations for a language"""
        with self.lock:
            if language not in self.translations:
                self.translations[language] = {}
            
            self.translations[language].update(translations)
            logger.info(f"Added {len(translations)} translations for language: {language}")
    
    def load_translations_from_file(self, filepath: str, language: str):
        """Load translations from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            self.add_translation(language, translations)
            logger.info(f"Loaded translations from {filepath} for language: {language}")
            
        except Exception as e:
            logger.error(f"Failed to load translations from {filepath}: {e}")
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation statistics"""
        with self.lock:
            stats = {
                'available_languages': len(self.translations),
                'current_language': self.current_language,
                'default_language': self.default_language,
                'languages': {}
            }
            
            for lang, translations in self.translations.items():
                stats['languages'][lang] = {
                    'translation_count': len(translations),
                    'coverage': len(translations) / max(1, len(self.translations[self.default_language]))
                }
            
            return stats


# Global localizer instance
_localizer = None

def get_localizer() -> Localizer:
    """Get global localizer instance"""
    global _localizer
    if _localizer is None:
        _localizer = Localizer()
    return _localizer


def translate(key: str, language: Optional[str] = None, **kwargs) -> str:
    """Convenience function for translation"""
    return get_localizer().translate(key, language, **kwargs)


def set_language(language: str) -> bool:
    """Convenience function to set language"""
    return get_localizer().set_language(language)


# Example usage and testing
if __name__ == "__main__":
    localizer = Localizer()
    
    # Test different languages
    for lang in ['en', 'es', 'fr', 'de', 'ja', 'zh']:
        localizer.set_language(lang)
        print(f"{lang}: {localizer.translate('platform.name')}")
        print(f"{lang}: {localizer.translate('discovery.process.started')}")
        print()
    
    # Test translation stats
    stats = localizer.get_translation_stats()
    print("Translation Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")