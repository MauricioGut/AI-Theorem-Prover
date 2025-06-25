"""
Manejo de datasets de teoremas separado
"""

import json
from pathlib import Path
from typing import List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TheoremExample:
    """Estructura para teoremas de entrenamiento"""
    premise: str
    conclusion: str
    proof_sketch: str
    domain: str  # algebra, logic, analysis, etc.
    difficulty: int  # 1-5

class TheoremDataset:
    """Manejo del dataset de teoremas"""
    
    def __init__(self, data_path: str = "data/theorems_dataset.json"):
        self.data_path = Path(data_path)
        self.examples: List[TheoremExample] = []
        self.load_or_create_dataset()
    
    def load_or_create_dataset(self):
        """Carga dataset existente o crea uno básico"""
        if self.data_path.exists():
            self.load_dataset()
        else:
            self.create_sample_dataset()
            self.save_dataset()
    
    def create_sample_dataset(self):
        """Crea un dataset de ejemplo con teoremas básicos"""
        sample_theorems = [
            TheoremExample(
                premise="∀x (P(x) → Q(x)) ∧ ∃x P(x)",
                conclusion="∃x Q(x)",
                proof_sketch="Modus ponens + existential instantiation",
                domain="logic",
                difficulty=2
            ),
            TheoremExample(
                premise="∀x (Mortal(x) ← Human(x)) ∧ Human(Socrates)",
                conclusion="Mortal(Socrates)",
                proof_sketch="Universal instantiation + modus ponens",
                domain="logic",
                difficulty=1
            ),
            # ... más ejemplos
        ]
        self.examples.extend(sample_theorems)
    
    def load_dataset(self):
        """Carga dataset desde archivo JSON"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'theorems' in data:
                    # Formato con metadata
                    theorem_data = data['theorems']
                else:
                    # Formato simple
                    theorem_data = data
                
                self.examples = [TheoremExample(**item) for item in theorem_data]
            logger.info(f"Cargados {len(self.examples)} teoremas")
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            self.create_sample_dataset()
    
    def save_dataset(self):
        """Guarda dataset en archivo JSON"""
        try:
            # Crear directorio si no existe
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.data_path, 'w', encoding='utf-8') as f:
                data = {
                    "metadata": {
                        "version": "1.0",
                        "total_theorems": len(self.examples)
                    },
                    "theorems": [vars(example) for example in self.examples]
                }
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Dataset guardado: {self.data_path}")
        except Exception as e:
            logger.error(f"Error guardando dataset: {e}")
    
    def get_training_texts(self) -> List[str]:
        """Prepara textos para entrenamiento"""
        texts = []
        for example in self.examples:
            # Formato: Premise → Conclusion
            text = f"Given: {example.premise}\nTherefore: {example.conclusion}"
            texts.append(text)
        return texts
