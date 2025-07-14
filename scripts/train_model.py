"""
Script para entrenar modelos de generación de teoremas

"""

import argparse
import logging
import sys
from pathlib import Path

# Añadir el directorio src al Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Ahora importar los módulos locales
try:
    from src.theorem_generator import TheoremGenerator
except ImportError as e:
    print(f"Error de importación: {e}")
    print(f"Asegúrate de que el archivo src/theorem_generator.py existe")
    print(f"Directorio actual: {Path.cwd()}")
    print(f"Buscando en: {src_path}")
    sys.exit(1)

def setup_logging():
    """Configurar logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo de generación de teoremas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos del modelo
    parser.add_argument('--model', 
                       default='microsoft/DialoGPT-medium',
                       help='Modelo base para fine-tuning')
    
    # Argumentos de entrenamiento
    parser.add_argument('--epochs', type=int, default=3,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Tamaño del batch')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Tasa de aprendizaje')
    
    # Argumentos de datos
    parser.add_argument('--dataset-path', 
                       default='data/theorems_dataset.json',
                       help='Ruta al dataset de teoremas')
    
    # Argumentos de salida
    parser.add_argument('--output-dir', 
                       default='models/trained_model',
                       help='Directorio de salida del modelo')
    parser.add_argument('--save-steps', type=int, default=500,
                       help='Frecuencia de guardado durante entrenamiento')
    
    # Argumentos opcionales
    parser.add_argument('--max-length', type=int, default=512,
                       help='Longitud máxima de secuencia')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Pasos de calentamiento')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Desactivar uso de GPU')
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Mostrar configuración
    logger.info("🔬 Iniciando entrenamiento de modelo de teoremas")
    logger.info("=" * 60)
    logger.info(f"Modelo base: {args.model}")
    logger.info(f"Épocas: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("-" * 60)
    
    try:
        # Verificar que el dataset existe
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            logger.warning(f"Dataset no encontrado en {dataset_path}")
            logger.info("Se creará un dataset de ejemplo")
        
        # Inicializar generador
        logger.info("Inicializando generador de teoremas...")
        generator = TheoremGenerator(model_name=args.model)
        
        # Configurar dataset si es necesario
        if hasattr(generator.dataset, 'data_path'):
            generator.dataset.data_path = Path(args.dataset_path)
            generator.dataset.load_or_create_dataset()
        
        logger.info(f"Dataset cargado: {len(generator.dataset.examples)} ejemplos")
        
        # Entrenar modelo
        logger.info("Iniciando fine-tuning...")
        generator.fine_tune(
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        
        # Crear directorio de salida
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar modelo entrenado
        logger.info(f"Guardando modelo en {output_path}...")
        generator.model.save_pretrained(output_path)
        generator.tokenizer.save_pretrained(output_path)
        
        # Guardar configuración de entrenamiento
        config_file = output_path / "training_config.json"
        import json
        config = {
            "base_model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dataset_path": str(args.dataset_path),
            "max_length": args.max_length,
            "warmup_steps": args.warmup_steps
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("✅ Entrenamiento completado exitosamente!")
        logger.info(f"Modelo guardado en: {output_path}")
        
        # Probar el modelo entrenado
        logger.info("\n🧪 Probando modelo entrenado...")
        test_premise = "∀x (P(x) → Q(x)) ∧ ∃x P(x)"
        results = generator.generate_theorem(test_premise, max_length=80)
        
        logger.info(f"Premisa de prueba: {test_premise}")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. {result}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Entrenamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()