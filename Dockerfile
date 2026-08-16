# Multi-stage Dockerfile for Safe-ICE.
#
# Build:  docker build -t safe-ice .
# Run:    docker run --rm -it safe-ice
#
# The package uses PEP 621 metadata, so pip installs it directly; there is no
# need to bring Poetry into the image just to resolve dependencies.

# ---------------------------------------------------------------- build stage
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Build into a self-contained virtualenv that the runtime stage can copy whole.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY safe_ice ./safe_ice

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# -------------------------------------------------------------- runtime stage
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash safeice

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
COPY --chown=safeice:safeice examples /workspace/examples

USER safeice

CMD ["safe-ice", "--help"]

LABEL org.opencontainers.image.title="Safe-ICE" \
      org.opencontainers.image.description="Safe Cross-Entropy-Based Importance Sampling for rare event simulation" \
      org.opencontainers.image.source="https://github.com/DiogoRibeiro7/adaptive-importance-sampling-ice" \
      org.opencontainers.image.licenses="MIT"

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import safe_ice" || exit 1
