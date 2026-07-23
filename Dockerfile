# Base image: lightweight Python 3.12 runtime (Debian slim).
FROM python:3.12-slim

# Copy the uv package manager binary from the official uv image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container for all following commands.
WORKDIR /app

# UV_COMPILE_BYTECODE: pre-compile .pyc files for faster startup.
# UV_LINK_MODE=copy: copy files into the venv instead of symlinking (safer in Docker).
# PYTHONUNBUFFERED: stream logs immediately (no buffering).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Copy dependency manifests first (Docker layer cache: only re-install if these change).
COPY pyproject.toml uv.lock ./

# Install locked production deps into .venv (skip dev group and this app as a package).
RUN uv sync --frozen --no-group dev --no-install-project

# Copy application source code into the image.
COPY service ./service

# Put the virtualenv on PATH so uvicorn runs without `uv run`.
ENV PATH="/app/.venv/bin:$PATH"

# Document that the container listens on port 8000 (does not publish the port by itself).
EXPOSE 8000

# Periodic health check: ECS/Docker can restart the container if /health fails.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Start the FastAPI app via uvicorn, bound to all interfaces on port 8000.
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
