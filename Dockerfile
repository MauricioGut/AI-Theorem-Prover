FROM python:3.9-slim-bookworm

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY setup.py .
COPY src/ ./src/
COPY data/ ./data/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["python", "-m", "theorem_generator"]