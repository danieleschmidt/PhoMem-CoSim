"""
Internationalization (i18n) support for PhoMem-CoSim.
Multi-language support for global deployment.
"""

import os
import json
import locale
import gettext
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import threading
from functools import wraps

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français', 
    'de': 'Deutsch',
    'ja': '日本語',
    'zh': '中文',
    'ko': '한국어',
    'ru': 'Русский',
    'pt': 'Português',
    'it': 'Italiano'
}

@dataclass
class LocaleInfo:
    """Locale information."""
    language_code: str
    country_code: Optional[str] = None
    encoding: str = 'utf-8'
    display_name: str = ''
    rtl: bool = False  # Right-to-left text direction
    
    @property
    def full_code(self) -> str:
        if self.country_code:
            return f"{self.language_code}_{self.country_code}"
        return self.language_code

class TranslationManager:
    """Manages translations and localization."""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent / 'locales'
        self.current_locale = LocaleInfo('en', 'US', display_name='English')
        self._translations = {}
        self._lock = threading.RLock()
        
        # Initialize with system locale if available
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                parts = system_locale.split('_')
                if len(parts) >= 1 and parts[0] in SUPPORTED_LANGUAGES:
                    self.current_locale.language_code = parts[0]
                    if len(parts) >= 2:
                        self.current_locale.country_code = parts[1]
        except:
            pass  # Use default English
        
        # Load default translations
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files."""
        with self._lock:
            locale_dir = self.base_dir / self.current_locale.language_code
            
            # Load JSON translation file
            json_file = locale_dir / 'messages.json'
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        self._translations = json.load(f)
                except Exception as e:
                    print(f"Failed to load translations from {json_file}: {e}")
            
            # Try gettext .mo file as fallback
            mo_file = locale_dir / 'LC_MESSAGES' / 'phomem.mo'
            if mo_file.exists():
                try:
                    translation = gettext.translation(
                        'phomem', 
                        str(self.base_dir),
                        languages=[self.current_locale.language_code]
                    )
                    # Convert gettext to our format if needed
                except Exception as e:
                    print(f"Failed to load gettext translations: {e}")
    
    def set_locale(self, language_code: str, country_code: Optional[str] = None) -> bool:
        """Set current locale."""
        if language_code not in SUPPORTED_LANGUAGES:
            return False
        
        with self._lock:
            old_locale = self.current_locale
            self.current_locale = LocaleInfo(
                language_code=language_code,
                country_code=country_code,
                display_name=SUPPORTED_LANGUAGES[language_code]
            )
            
            # Set RTL for Arabic, Hebrew, etc.
            rtl_languages = {'ar', 'he', 'fa', 'ur'}
            self.current_locale.rtl = language_code in rtl_languages
            
            # Load new translations
            self._load_translations()
            
            # Set system locale if possible
            try:
                locale_str = self.current_locale.full_code + '.UTF-8'
                locale.setlocale(locale.LC_ALL, locale_str)
            except:
                try:
                    # Try without country code
                    locale.setlocale(locale.LC_ALL, language_code + '.UTF-8')
                except:
                    pass  # Keep current system locale
            
            return True
    
    def get_text(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """Get translated text with optional formatting."""
        with self._lock:
            text = self._translations.get(key, default or key)
            
            # Format with provided arguments
            if kwargs:
                try:
                    text = text.format(**kwargs)
                except (KeyError, ValueError):
                    pass  # Return unformatted text
            
            return text
    
    def get_plural(self, key: str, count: int, default: Optional[str] = None) -> str:
        """Get pluralized text based on count."""
        with self._lock:
            plural_key = f"{key}_plural"
            
            # Simple English pluralization rules
            if self.current_locale.language_code == 'en':
                if count == 1:
                    return self.get_text(key, default)
                else:
                    return self.get_text(plural_key, default or f"{key}s")
            
            # Other languages may have different pluralization rules
            # This is a simplified implementation
            return self.get_text(plural_key if count != 1 else key, default)
    
    def format_number(self, number: Union[int, float], decimal_places: int = 2) -> str:
        """Format number according to locale."""
        try:
            if isinstance(number, int):
                return f"{number:,}"
            else:
                return f"{number:,.{decimal_places}f}"
        except:
            return str(number)
    
    def format_currency(self, amount: float, currency_code: str = 'USD') -> str:
        """Format currency according to locale."""
        try:
            # Simple currency formatting - would use babel in production
            currency_symbols = {
                'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
                'CNY': '¥', 'KRW': '₩', 'RUB': '₽'
            }
            
            symbol = currency_symbols.get(currency_code, currency_code)
            formatted_amount = self.format_number(amount, 2)
            
            # Different positioning for different locales
            if self.current_locale.language_code in ['de', 'fr']:
                return f"{formatted_amount} {symbol}"
            else:
                return f"{symbol}{formatted_amount}"
        except:
            return f"{amount} {currency_code}"
    
    def get_locale_info(self) -> LocaleInfo:
        """Get current locale information."""
        return self.current_locale
    
    def get_available_locales(self) -> List[LocaleInfo]:
        """Get list of available locales."""
        locales = []
        for code, name in SUPPORTED_LANGUAGES.items():
            locale_info = LocaleInfo(
                language_code=code,
                display_name=name,
                rtl=code in {'ar', 'he', 'fa', 'ur'}
            )
            locales.append(locale_info)
        return locales

class RegionalCompliance:
    """Handle regional compliance requirements (GDPR, CCPA, etc.)."""
    
    def __init__(self):
        self.compliance_rules = {
            'gdpr': {  # General Data Protection Regulation (EU)
                'regions': ['EU', 'EEA'],
                'data_retention_days': 365,
                'consent_required': True,
                'right_to_deletion': True,
                'data_portability': True,
                'privacy_by_design': True
            },
            'ccpa': {  # California Consumer Privacy Act
                'regions': ['CA', 'US-CA'],
                'data_retention_days': 365,
                'consent_required': True,
                'right_to_deletion': True,
                'data_portability': True,
                'opt_out_rights': True
            },
            'pdpa': {  # Personal Data Protection Act (Singapore)
                'regions': ['SG'],
                'data_retention_days': 365,
                'consent_required': True,
                'notification_required': True
            },
            'pipeda': {  # Personal Information Protection (Canada)
                'regions': ['CA'],
                'data_retention_days': 365,
                'consent_required': True,
                'right_to_access': True
            }
        }
    
    def get_applicable_regulations(self, region_code: str) -> List[str]:
        """Get applicable regulations for a region."""
        applicable = []
        
        for regulation, rules in self.compliance_rules.items():
            if region_code in rules['regions']:
                applicable.append(regulation)
        
        return applicable
    
    def check_data_retention(self, region_code: str, retention_days: int) -> Dict[str, Any]:
        """Check if data retention complies with regulations."""
        applicable_regs = self.get_applicable_regulations(region_code)
        
        compliance_status = {
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        for reg in applicable_regs:
            rules = self.compliance_rules[reg]
            max_retention = rules.get('data_retention_days', 365)
            
            if retention_days > max_retention:
                compliance_status['compliant'] = False
                compliance_status['violations'].append(
                    f"{reg.upper()}: Data retention exceeds {max_retention} days"
                )
                compliance_status['recommendations'].append(
                    f"Reduce data retention to {max_retention} days or less"
                )
        
        return compliance_status
    
    def get_privacy_requirements(self, region_code: str) -> Dict[str, Any]:
        """Get privacy requirements for a region."""
        applicable_regs = self.get_applicable_regulations(region_code)
        
        requirements = {
            'consent_required': False,
            'right_to_deletion': False,
            'data_portability': False,
            'privacy_by_design': False,
            'opt_out_rights': False,
            'right_to_access': False,
            'notification_required': False,
            'applicable_regulations': applicable_regs
        }
        
        for reg in applicable_regs:
            rules = self.compliance_rules[reg]
            for key in requirements:
                if key in rules and rules[key]:
                    requirements[key] = True
        
        return requirements

class CrossPlatformSupport:
    """Handle cross-platform compatibility and adaptations."""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detect current platform information."""
        import platform
        import sys
        
        info = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': sys.version,
            'is_windows': platform.system() == 'Windows',
            'is_mac': platform.system() == 'Darwin',
            'is_linux': platform.system() == 'Linux',
            'is_64bit': platform.machine().endswith('64'),
            'path_separator': os.sep,
            'line_separator': os.linesep
        }
        
        return info
    
    def get_platform_specific_path(self, base_path: str) -> str:
        """Get platform-specific file path."""
        if self.platform_info['is_windows']:
            # Windows-specific path handling
            return base_path.replace('/', '\\')
        else:
            # Unix-like systems
            return base_path.replace('\\', '/')
    
    def get_data_directory(self) -> Path:
        """Get platform-appropriate data directory."""
        if self.platform_info['is_windows']:
            base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        elif self.platform_info['is_mac']:
            base = Path.home() / 'Library' / 'Application Support'
        else:
            # Linux/Unix
            base = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
        
        return base / 'PhoMem-CoSim'
    
    def get_config_directory(self) -> Path:
        """Get platform-appropriate config directory."""
        if self.platform_info['is_windows']:
            base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        elif self.platform_info['is_mac']:
            base = Path.home() / 'Library' / 'Preferences'
        else:
            # Linux/Unix
            base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
        
        return base / 'phomem'
    
    def get_temp_directory(self) -> Path:
        """Get platform-appropriate temporary directory."""
        import tempfile
        return Path(tempfile.gettempdir()) / 'phomem'
    
    def create_directories(self):
        """Create necessary directories with proper permissions."""
        dirs_to_create = [
            self.get_data_directory(),
            self.get_config_directory(),
            self.get_temp_directory()
        ]
        
        for directory in dirs_to_create:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                
                # Set appropriate permissions on Unix-like systems
                if not self.platform_info['is_windows']:
                    os.chmod(directory, 0o755)
                    
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {e}")

# Global instances
_translation_manager = TranslationManager()
_compliance_manager = RegionalCompliance()
_platform_manager = CrossPlatformSupport()

def get_translation_manager() -> TranslationManager:
    """Get global translation manager."""
    return _translation_manager

def get_compliance_manager() -> RegionalCompliance:
    """Get global compliance manager."""
    return _compliance_manager

def get_platform_manager() -> CrossPlatformSupport:
    """Get global platform manager."""
    return _platform_manager

# Convenience functions
def _(text: str, **kwargs) -> str:
    """Shorthand for getting translated text."""
    return _translation_manager.get_text(text, **kwargs)

def ngettext(singular: str, plural: str, count: int) -> str:
    """Get pluralized text."""
    return _translation_manager.get_plural(singular, count, plural)

def set_language(language_code: str, country_code: Optional[str] = None) -> bool:
    """Set application language."""
    return _translation_manager.set_locale(language_code, country_code)

def format_number(number: Union[int, float], decimal_places: int = 2) -> str:
    """Format number for current locale."""
    return _translation_manager.format_number(number, decimal_places)

def format_currency(amount: float, currency_code: str = 'USD') -> str:
    """Format currency for current locale."""
    return _translation_manager.format_currency(amount, currency_code)

def check_compliance(region_code: str) -> Dict[str, Any]:
    """Check compliance requirements for region."""
    return _compliance_manager.get_privacy_requirements(region_code)

def get_platform_info() -> Dict[str, Any]:
    """Get current platform information."""
    return _platform_manager.platform_info

# Decorator for internationalized functions
def internationalized(func):
    """Decorator to mark functions as internationalized."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper._internationalized = True
    return wrapper

# Initialize platform directories
try:
    _platform_manager.create_directories()
except Exception as e:
    print(f"Warning: Platform initialization failed: {e}")

# Default translations for common terms
DEFAULT_TRANSLATIONS = {
    'en': {
        'error': 'Error',
        'warning': 'Warning',
        'success': 'Success',
        'loading': 'Loading...',
        'save': 'Save',
        'cancel': 'Cancel',
        'ok': 'OK',
        'yes': 'Yes',
        'no': 'No',
        'simulation_complete': 'Simulation Complete',
        'training_complete': 'Training Complete',
        'file_not_found': 'File not found',
        'invalid_input': 'Invalid input',
        'network_error': 'Network error',
        'device_error': 'Device error',
        'optimization_failed': 'Optimization failed'
    },
    'es': {
        'error': 'Error',
        'warning': 'Advertencia',
        'success': 'Éxito',
        'loading': 'Cargando...',
        'save': 'Guardar',
        'cancel': 'Cancelar',
        'ok': 'Aceptar',
        'yes': 'Sí',
        'no': 'No',
        'simulation_complete': 'Simulación Completa',
        'training_complete': 'Entrenamiento Completo',
        'file_not_found': 'Archivo no encontrado',
        'invalid_input': 'Entrada inválida',
        'network_error': 'Error de red',
        'device_error': 'Error del dispositivo',
        'optimization_failed': 'Optimización fallida'
    },
    'fr': {
        'error': 'Erreur',
        'warning': 'Avertissement', 
        'success': 'Succès',
        'loading': 'Chargement...',
        'save': 'Enregistrer',
        'cancel': 'Annuler',
        'ok': 'OK',
        'yes': 'Oui',
        'no': 'Non',
        'simulation_complete': 'Simulation Terminée',
        'training_complete': 'Entraînement Terminé',
        'file_not_found': 'Fichier non trouvé',
        'invalid_input': 'Entrée invalide',
        'network_error': 'Erreur réseau',
        'device_error': 'Erreur de périphérique',
        'optimization_failed': 'Optimisation échouée'
    },
    'de': {
        'error': 'Fehler',
        'warning': 'Warnung',
        'success': 'Erfolg',
        'loading': 'Wird geladen...',
        'save': 'Speichern',
        'cancel': 'Abbrechen',
        'ok': 'OK',
        'yes': 'Ja',
        'no': 'Nein',
        'simulation_complete': 'Simulation Abgeschlossen',
        'training_complete': 'Training Abgeschlossen',
        'file_not_found': 'Datei nicht gefunden',
        'invalid_input': 'Ungültige Eingabe',
        'network_error': 'Netzwerkfehler',
        'device_error': 'Gerät Fehler',
        'optimization_failed': 'Optimierung fehlgeschlagen'
    },
    'ja': {
        'error': 'エラー',
        'warning': '警告',
        'success': '成功',
        'loading': '読み込み中...',
        'save': '保存',
        'cancel': 'キャンセル',
        'ok': 'OK',
        'yes': 'はい',
        'no': 'いいえ',
        'simulation_complete': 'シミュレーション完了',
        'training_complete': 'トレーニング完了',
        'file_not_found': 'ファイルが見つかりません',
        'invalid_input': '無効な入力',
        'network_error': 'ネットワークエラー',
        'device_error': 'デバイスエラー',
        'optimization_failed': '最適化失敗'
    },
    'zh': {
        'error': '错误',
        'warning': '警告',
        'success': '成功',
        'loading': '加载中...',
        'save': '保存',
        'cancel': '取消',
        'ok': '确定',
        'yes': '是',
        'no': '否',
        'simulation_complete': '仿真完成',
        'training_complete': '训练完成',
        'file_not_found': '文件未找到',
        'invalid_input': '无效输入',
        'network_error': '网络错误',
        'device_error': '设备错误',
        'optimization_failed': '优化失败'
    }
}

# Update translation manager with default translations
_translation_manager._translations.update(
    DEFAULT_TRANSLATIONS.get(_translation_manager.current_locale.language_code, DEFAULT_TRANSLATIONS['en'])
)