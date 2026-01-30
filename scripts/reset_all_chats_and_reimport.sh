#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${1:-}"
DEFAULT_PLATFORM="${2:-taobao}"
DRY_RUN="${3:-}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found" >&2
  exit 1
fi

docker compose up -d --build >/dev/null

ARGS=(python -m tools.reset_chat_and_reimport_from_backup --default-platform "$DEFAULT_PLATFORM")
if [[ -n "$BACKUP_DIR" ]]; then
  ARGS+=(--backup-dir "$BACKUP_DIR")
fi
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  ARGS+=(--dry-run)
fi

docker compose exec -T app "${ARGS[@]}"
