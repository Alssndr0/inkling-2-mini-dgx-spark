#!/usr/bin/env bash
# Download thinkingmachines/Inkling-2-Mini at the pinned revision and verify every file against the values Hugging
# Face publishes for that revision (checksums/Inkling-2-Mini-b85c3cf.txt: the Git LFS SHA-256 of each large file,
# the Git blob SHA-1 of each small one). Resumable; re-run it to finish or re-check a download.
#
#   ./download.sh                 -> ./models/Inkling-2-Mini (about 20 GB)
#   MODEL_DIR=/data/inkling ./download.sh
#   HF_TOKEN=hf_... ./download.sh   (only if your account needs a token for this repository)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO=thinkingmachines/Inkling-2-Mini
REV=b85c3cfa615fc413cc9af3379f491b963f3b586a
LIST="$HERE/checksums/Inkling-2-Mini-b85c3cf.txt"
DST="${MODEL_DIR:-$HERE/models/Inkling-2-Mini}"
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
mkdir -p "$DST"

check() {  # path size kind hash -> 0 if the local file matches
  local f="$DST/$1"
  [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$2" ] || return 1
  if [ "$3" = sha256 ]; then
    [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$4" ]
  else
    [ "$( { printf 'blob %s\0' "$2"; cat "$f"; } | sha1sum | cut -d' ' -f1)" = "$4" ]
  fi
}

bad=0
while read -r path size kind hash; do
  case "$path" in ''|'#'*) continue ;; esac
  if check "$path" "$size" "$kind" "$hash"; then echo "ok        $path"; continue; fi
  mkdir -p "$(dirname "$DST/$path")"
  for i in 1 2 3 4 5; do
    curl -fL -# --retry 3 -C - "${AUTH[@]}" -o "$DST/$path" "https://huggingface.co/$REPO/resolve/$REV/$path" && break || true
    echo "retry $path ($i)"; sleep $((i * 5))
  done
  if check "$path" "$size" "$kind" "$hash"; then echo "verified  $path"; else echo "MISMATCH  $path" >&2; bad=1; fi
done < "$LIST"
[ "$bad" = 0 ] || { echo "some files do not match the pinned revision: delete them and re-run" >&2; exit 1; }
echo "$REV" > "$DST/REVISION"
echo "all files verified: $DST (revision $REV)"
