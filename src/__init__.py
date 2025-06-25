"""
AI Theorem Prover Package
Generador automático de teoremas usando redes neuronales
"""

__version__ = "0.1.0"
__author__ = "Mauricio Gutierrez"
__email__ = "mauriciogut67@gmail.com"

# Importaciones principales
from .theorem_generator import TheoremGenerator

# Importaciones opcionales (solo si existen los archivos)
try:
    from .logic_validator import LogicValidator
    from .dataset_manager import TheoremDataset
except ImportError:
    # Si no existen los archivos modulares, usar las clases del archivo principal
    from .theorem_generator import LogicValidator, TheoremDataset

__all__ = [
    "TheoremGenerator",
    "LogicValidator", 
    "TheoremDataset"
]

# Información del paquete
def get_version():
    """Obtiene la versión del paquete"""
    return __version__

def get_info():
    """Obtiene información del paquete"""
    return {
        "name": "AI Theorem Prover",
        "version": __version__,
        "author": __author__,
        "description": "Generador automático de teoremas usando ML"
    }
