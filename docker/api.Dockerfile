FROM python:3.12-slim

WORKDIR /app

# Системные зависимости для работы с PDF и документами
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    antiword \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

RUN mkdir -p /app/uploads /app/logs

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
