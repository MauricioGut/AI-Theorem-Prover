FROM python:3.9-slim-bullseye

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos
COPY requirements.txt .
COPY setup.py .
COPY src/ ./src/
COPY data/ ./data/

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Exponer puerto para interfaz web
EXPOSE 8000

# Comando por defecto
CMD ["python", "-m", "theorem_generator"]