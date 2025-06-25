#!/usr/bin/env python3
"""
Script para entrenar modelos de generación de teoremas
"""

import argparse
import logging
from pathlib import Path
from theorem_generator import TheoremGenerator

def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo de teoremas')
    parser.add_argument('--model', default='microsoft/DialoGPT-medium',
                       help='Modelo base para fine-tuning')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Tamaño del batch')
    parser.add_argument('--output-dir', default='./models/trained_model',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Inicializar y entrenar
    generator = TheoremGenerator(args.model)
    generator.fine_tune(epochs=args.epochs, batch_size=args.batch_size)
    
    # Guardar modelo
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generator.model.save_pretrained(output_path)
    generator.tokenizer.save_pretrained(output_path)
    
    print(f"Modelo guardado en: {output_path}")

if __name__ == "__main__":
    main()