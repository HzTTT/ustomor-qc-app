FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Codex CLI runs inside the worker/app containers (Marllen assistant upstream=codex).
# We install Node + Codex CLI + a couple of common utilities Codex often uses.
ARG NODE_VERSION=18.20.4
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      ripgrep \
      xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && set -eux; \
      arch="${TARGETARCH:-}"; \
      if [ -z "$arch" ]; then arch="$(dpkg --print-architecture)"; fi; \
      if [ "$arch" = "amd64" ] || [ "$arch" = "x86_64" ]; then node_arch="x64"; \
      elif [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then node_arch="arm64"; \
      else echo "Unsupported arch: $arch" >&2; exit 1; fi; \
      curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" \
        | tar -xJ -C /usr/local --strip-components=1; \
      npm config set fund false; \
      npm config set audit false; \
      npm install -g @openai/codex@0.99.0-alpha.5

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
