# Use the official uv image as base
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Install system dependencies required for vLLM and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies (this layer will be cached unless pyproject.toml changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p rlwot && \
    touch rlwot/__init__.py && \
    uv pip install --system -e .

RUN <<EOF cat >> /etc/security/limits.conf
* soft memlock unlimited
* hard memlock unlimited
* soft core unlimited
* hard core unlimited
* soft nofile unlimited
* hard nofile unlimited
EOF

# Copy source code (changes here won't invalidate dependency layer)
COPY rlwot ./rlwot/

# Verify installation
RUN python -c "import rlwot; print(f'RLWOT version: {rlwot.__version__}')" && \
    which rlwot_es_naive

# Set the default command
CMD ["rlwot_es_naive", "--help"]
