FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

WORKDIR /app 

COPY pyproject.toml uv.lock* ./ 

RUN uv pip install --system -r pyproject.toml 

COPY src/ ./src/ 

CMD ["python", "src/evaluation/main.py"]
