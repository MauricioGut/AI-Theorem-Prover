"""
Generador Automático de Teoremas usando Redes Neuronales
Implementación avanzada con múltiples enfoques y validación lógica
"""

import json
import re
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Dependencias principales
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        pipeline, Trainer, TrainingArguments
    )
    import torch
    from datasets import Dataset
    import numpy as np
    from sympy.logic import simplify_logic
    from sympy.parsing.sympy_parser import parse_expr
except ImportError as e:
    print(f"Instalar dependencias: pip install transformers torch datasets sympy")
    raise e

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TheoremExample:
    """Estructura para teoremas de entrenamiento"""
    premise: str
    conclusion: str
    proof_sketch: str
    domain: str  # algebra, logic, analysis, etc.
    difficulty: int  # 1-5

class LogicValidator:
    """Validador básico de lógica formal"""
    
    def __init__(self):
        # Patrones de lógica proposicional y predicados
        self.logic_patterns = {
            'forall': r'∀[a-zA-Z]\s*\(',
            'exists': r'∃[a-zA-Z]\s*\(',
            'implication': r'→',
            'conjunction': r'∧',
            'disjunction': r'∨',
            'negation': r'¬',
            'biconditional': r'↔'
        }
    
    def is_valid_formula(self, formula: str) -> bool:
        """Validación básica de sintaxis lógica"""
        try:
            # Verificar balance de paréntesis
            if formula.count('(') != formula.count(')'):
                return False
            
            # Verificar presencia de conectores lógicos válidos
            has_logic = any(re.search(pattern, formula) 
                          for pattern in self.logic_patterns.values())
            
            return has_logic and len(formula.strip()) > 0
        except:
            return False
    
    def extract_predicates(self, formula: str) -> List[str]:
        """Extrae predicados de una fórmula"""
        predicates = re.findall(r'[A-Z]\([a-zA-Z,\s]*\)', formula)
        return list(set(predicates))

class TheoremDataset:
    """Manejo del dataset de teoremas"""
    
    def __init__(self, data_path: str = "theorems_dataset.json"):
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
            TheoremExample(
                premise="∀x ∀y (R(x,y) → R(y,x)) ∧ ∀x R(x,x)",
                conclusion="∀x ∀y (R(x,y) ↔ R(y,x))",
                proof_sketch="Symmetry + reflexivity implies equivalence",
                domain="relations",
                difficulty=3
            ),
            TheoremExample(
                premise="∀x (Prime(x) ∧ x > 2 → Odd(x))",
                conclusion="∀x (Prime(x) ∧ Even(x) → x = 2)",
                proof_sketch="Contrapositive of primality condition",
                domain="number_theory",
                difficulty=3
            ),
            TheoremExample(
                premise="∀x ∀y (x < y → ∃z (x < z ∧ z < y))",
                conclusion="¬∃x ∃y (x < y ∧ ∀z ¬(x < z ∧ z < y))",
                proof_sketch="Dense ordering property",
                domain="analysis",
                difficulty=4
            )
        ]
        self.examples.extend(sample_theorems)
    
    def load_dataset(self):
        """Carga dataset desde archivo JSON"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.examples = [TheoremExample(**item) for item in data]
            logger.info(f"Cargados {len(self.examples)} teoremas")
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            self.create_sample_dataset()
    
    def save_dataset(self):
        """Guarda dataset en archivo JSON"""
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                data = [vars(example) for example in self.examples]
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

class TheoremGenerator:
    """Generador principal de teoremas"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.validator = LogicValidator()
        self.dataset = TheoremDataset()
        
        # Cargar modelo
        self.load_model()
    
    def load_model(self):
        """Carga el modelo y tokenizer"""
        try:
            logger.info(f"Cargando modelo: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Añadir token de padding si no existe
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def fine_tune(self, epochs: int = 3, batch_size: int = 4):
        """Fine-tuning del modelo con dataset de teoremas"""
        try:
            logger.info("Iniciando fine-tuning...")
            
            # Preparar datos
            texts = self.dataset.get_training_texts()
            
            # Tokenizar
            tokenized = self.tokenizer(
                texts, 
                truncation=True, 
                padding=True, 
                max_length=512,
                return_tensors="pt"
            )
            
            # Crear dataset
            train_dataset = Dataset.from_dict({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': tokenized['input_ids'].clone()
            })
            
            # Configurar entrenamiento
            training_args = TrainingArguments(
                output_dir='./theorem_model',
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=1000,
                save_total_limit=2,
                logging_steps=100,
                learning_rate=5e-5,
                warmup_steps=100,
            )
            
            # Entrenar
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
            )
            
            trainer.train()
            logger.info("Fine-tuning completado")
            
        except Exception as e:
            logger.error(f"Error en fine-tuning: {e}")
    
    def generate_theorem(self, premise: str, max_length: int = 100, 
                        num_return_sequences: int = 3) -> List[str]:
        """Genera teoremas basados en una premisa"""
        try:
            # Validar premisa
            if not self.validator.is_valid_formula(premise):
                logger.warning(f"Premisa posiblemente inválida: {premise}")
            
            # Preparar prompt
            prompt = f"Given: {premise}\nTherefore:"
            
            # Tokenizar
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generar
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decodificar resultados
            results = []
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Extraer solo la conclusión
                if "Therefore:" in text:
                    conclusion = text.split("Therefore:")[-1].strip()
                    if conclusion and self.validator.is_valid_formula(conclusion):
                        results.append(conclusion)
            
            return results
            
        except Exception as e:
            logger.error(f"Error generando teorema: {e}")
            return []
    
    def generate_conjecture(self, domain: str = "logic") -> Dict[str, str]:
        """Genera una conjetura en un dominio específico"""
        # Filtrar ejemplos por dominio
        domain_examples = [ex for ex in self.dataset.examples 
                          if ex.domain == domain]
        
        if not domain_examples:
            domain_examples = self.dataset.examples
        
        # Seleccionar ejemplo base aleatorio
        base_example = random.choice(domain_examples)
        
        # Generar variación
        variations = self.generate_theorem(base_example.premise, max_length=80)
        
        if variations:
            return {
                'premise': base_example.premise,
                'conjecture': variations[0],
                'domain': domain,
                'confidence': self._calculate_confidence(variations[0])
            }
        
        return {'error': 'No se pudo generar conjetura'}
    
    def _calculate_confidence(self, formula: str) -> float:
        """Calcula confianza basada en validez sintáctica"""
        score = 0.0
        
        # Validez sintáctica
        if self.validator.is_valid_formula(formula):
            score += 0.5
        
        # Complejidad apropiada
        predicates = self.validator.extract_predicates(formula)
        if 1 <= len(predicates) <= 5:
            score += 0.3
        
        # Presencia de conectores lógicos
        logic_count = sum(1 for pattern in self.validator.logic_patterns.values()
                         if re.search(pattern, formula))
        if logic_count > 0:
            score += 0.2
        
        return min(score, 1.0)

def main():
    """Función principal de demostración"""
    print("🔬 Generador Automático de Teoremas")
    print("=" * 50)
    
    # Inicializar generador
    generator = TheoremGenerator()
    
    # Ejemplos de uso
    test_premises = [
        "∀x (P(x) → Q(x)) ∧ ∃x P(x)",
        "∀x (Student(x) → Person(x)) ∧ Student(Alice)",
        "∀x ∀y (Loves(x,y) → Knows(x,y))"
    ]
    
    print("\n📝 Generando teoremas...")
    for premise in test_premises:
        print(f"\nPremisa: {premise}")
        conclusions = generator.generate_theorem(premise, max_length=80)
        
        for i, conclusion in enumerate(conclusions, 1):
            confidence = generator._calculate_confidence(conclusion)
            print(f"  {i}. {conclusion} (confianza: {confidence:.2f})")
    
    print("\n🎯 Generando conjeturas por dominio...")
    domains = ["logic", "number_theory", "analysis"]
    
    for domain in domains:
        conjecture = generator.generate_conjecture(domain)
        if 'error' not in conjecture:
            print(f"\nDominio: {domain}")
            print(f"Premisa: {conjecture['premise']}")
            print(f"Conjetura: {conjecture['conjecture']}")
            print(f"Confianza: {conjecture['confidence']:.2f}")
    
    print("\n✅ Demostración completada")

if __name__ == "__main__":
    main()