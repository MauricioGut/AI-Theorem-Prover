"""
Validador de lógica formal separado del módulo principal
"""

import re
from typing import List

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