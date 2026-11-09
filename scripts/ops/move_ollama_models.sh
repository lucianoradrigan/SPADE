#!/usr/bin/env bash
# Move the Ollama SERVICE's model store to /mnt/datos/ollama/models (decision C9, round C, 2026-11-09).
#
#   sudo bash scripts/ops/move_ollama_models.sh            # do it (idempotent: safe to run again)
#   bash scripts/ops/move_ollama_models.sh --dry-run       # print what it would do; changes nothing, no sudo
#
# Steps: detect the store the service really uses -> check space on /mnt/datos -> stop ollama -> rsync -a
# -> verify the copy -> systemd override (OLLAMA_MODELS, RequiresMountsFor) -> daemon-reload + restart ->
# check that `ollama list` shows qwen3.5:9b with the pinned digest -> rename the old store to models.bak
# (NOT deleted; Luciano deletes it by hand).
#
# Rollback: sudo rm /etc/systemd/system/ollama.service.d/spade-models.conf && sudo mv <old>.bak <old>
#           && sudo systemctl daemon-reload && sudo systemctl restart ollama
set -euo pipefail

TARGET=/mnt/datos/ollama/models
MOUNT=/mnt/datos
OVERRIDE_DIR=/etc/systemd/system/ollama.service.d
OVERRIDE=$OVERRIDE_DIR/spade-models.conf
MODEL=qwen3.5:9b
DIGEST=56671c2ab9385f9cfcb404638e32cd62d88e3501d44822208363c010179a3c90  # configs/assistant.yaml
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

say() { printf '[move_ollama_models] %s\n' "$*"; }
die() { say "ERROR: $*"; exit 1; }

# 1. The store the service uses now: OLLAMA_MODELS in its environment, else <home of user ollama>/.ollama/models.
svc_env=$(systemctl show ollama -p Environment --value 2>/dev/null || true)
current=$(printf '%s\n' $svc_env | sed -n 's/^OLLAMA_MODELS=//p' | tail -1)
svc_user=$(systemctl show ollama -p User --value 2>/dev/null || true); svc_user=${svc_user:-ollama}
home=$(getent passwd "$svc_user" | cut -d: -f6)
SRC=${current:-$home/.ollama/models}
say "service user: $svc_user; store in use: $SRC; target: $TARGET"

if [ "$SRC" = "$TARGET" ]; then
  say "the service already uses $TARGET: nothing to move, only checking"
  ALREADY=1
else
  ALREADY=0
  [ -d "$SRC" ] || die "store $SRC not found"
  mountpoint -q "$MOUNT" || die "$MOUNT is not mounted"
  need=$(du -sb "$SRC" | cut -f1)
  have=$(df -B1 --output=avail "$MOUNT" | tail -1 | tr -d ' ')
  say "size of the store: $((need / 1024 / 1024)) MiB; free on $MOUNT: $((have / 1024 / 1024)) MiB"
  [ "$have" -gt $((need + need / 10)) ] || die "not enough space on $MOUNT (need the store + 10 %)"
fi

if [ "$DRY" -eq 1 ]; then
  say "dry run: would stop ollama, rsync -a $SRC/ -> $TARGET/, chown -R $svc_user, write $OVERRIDE,"
  say "         daemon-reload, restart, check '$MODEL' and its digest, then mv $SRC ${SRC%/}.bak"
  exit 0
fi
[ "$(id -u)" -eq 0 ] || die "run it with sudo (or use --dry-run)"

if [ "$ALREADY" -eq 0 ]; then
  say "stopping ollama"; systemctl stop ollama
  mkdir -p "$TARGET"
  say "copying (rsync -a) ..."; rsync -a "$SRC"/ "$TARGET"/
  chown -R "$svc_user":"$(id -gn "$svc_user")" "$TARGET"
  diffs=$(rsync -a --checksum --dry-run --itemize-changes "$SRC"/ "$TARGET"/ | wc -l)
  [ "$diffs" -eq 0 ] || die "the copy differs from the source in $diffs entries; the service was NOT switched (start it again with: systemctl start ollama)"
  say "copy verified (rsync --checksum: no differences)"
fi

mkdir -p "$OVERRIDE_DIR"
want=$(printf '[Unit]\nRequiresMountsFor=%s\n\n[Service]\nEnvironment="OLLAMA_MODELS=%s"\n' "$MOUNT" "$TARGET")
if [ ! -f "$OVERRIDE" ] || [ "$(cat "$OVERRIDE")" != "$want" ]; then
  printf '%s\n' "$want" > "$OVERRIDE"; say "wrote $OVERRIDE"
fi
systemctl daemon-reload
systemctl restart ollama

for _ in $(seq 1 30); do curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
curl -fsS http://127.0.0.1:11434/api/tags >/dev/null || die "the service does not answer on 127.0.0.1:11434"
ollama list | grep -q "^$MODEL" || die "'ollama list' does not show $MODEL (the old store is untouched at $SRC)"
curl -fsS http://127.0.0.1:11434/api/tags | grep -q "$DIGEST" || die "$MODEL is listed but its digest is not $DIGEST"
say "ollama list shows $MODEL with the pinned digest"

if [ "$ALREADY" -eq 0 ]; then
  if [ -e "${SRC%/}.bak" ]; then
    say "${SRC%/}.bak already exists: the old store stays at $SRC (not renamed)"
  else
    mv "$SRC" "${SRC%/}.bak"; say "old store renamed to ${SRC%/}.bak (not deleted)"
  fi
fi
say "done. Free space now:"; df -h / "$MOUNT"
