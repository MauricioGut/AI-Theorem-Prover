#!/bin/bash
# Script para diagnosticar y corregir errores de importación

echo "🔍 Diagnóstico del proyecto AI-Theorem-Prover"
echo "=============================================="

# 1. Verificar estructura actual
echo "📁 Estructura actual del proyecto:"
find . -type f -name "*.py" | head -20

echo -e "\n📄 Contenido del directorio src/:"
ls -la src/ 2>/dev/null || echo "❌ Directorio src/ no existe"

echo -e "\n📄 Primeras líneas de src/theorem_generator.py:"
head -10 src/theorem_generator.py 2>/dev/null || echo "❌ Archivo src/theorem_generator.py no existe o está vacío"

# 2. Verificar si el archivo tiene la clase TheoremGenerator
echo -e "\n🔍 Buscando clase TheoremGenerator:"
grep -n "class TheoremGenerator" src/theorem_generator.py 2>/dev/null || echo "❌ Clase TheoremGenerator no encontrada"

# 3. Verificar Python path
echo -e "\n🐍 Python path actual:"
python3 -c "import sys; print('\n'.join(sys.path))"

# 4. Verificar imports
echo -e "\n📦 Probando importación directa:"
python3 -c "
import sys
sys.path.insert(0, './src')
try:
    from theorem_generator import TheoremGenerator
    print('✅ Importación exitosa')
except ImportError as e:
    print(f'❌ Error de importación: {e}')
except Exception as e:
    print(f'❌ Error general: {e}')
"
