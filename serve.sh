#!/usr/bin/env bash
# Inkling 2 Mini on one DGX Spark (GB10): vLLM with this recipe's kernel overlays.
#
#   ./serve.sh [fastest|fast|lossless]   start a profile (default: fastest) and wait until it answers
#   ./serve.sh stop                      drain running requests, then stop the container
#   DRY_RUN=1 ./serve.sh fast            print the docker command instead of running it
#
# Settings (environment):
#   PORT=8000          host port of the OpenAI-compatible API
#   BIND=127.0.0.1     host address the port is published on (0.0.0.0 to serve your LAN)
#   MODEL_DIR=./models/Inkling-2-Mini   weights from ./download.sh
#   CACHE_DIR=./cache  persistent Triton / torch / FlashInfer caches (startup is ~60-90 s warm, longer cold)
#   CONTAINER=inkling  container name
#   KV_CACHE=64G       KV cache pool (vLLM --kv-cache-memory-bytes); 64G holds ~2.4 full-length requests
#   GPU_UTIL=0.85      vLLM --gpu-memory-utilization
#   DRAIN_S=120        seconds `stop` waits for running requests to finish
#
# Every overlay file is checked against patches/SHA256SUMS before launch and mounted read-only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e"
PORT="${PORT:-8000}"; BIND="${BIND:-127.0.0.1}"; CONTAINER="${CONTAINER:-inkling}"
MODEL_DIR="${MODEL_DIR:-$HERE/models/Inkling-2-Mini}"; CACHE_DIR="${CACHE_DIR:-$HERE/cache}"
KV_CACHE="${KV_CACHE:-64G}"; GPU_UTIL="${GPU_UTIL:-0.85}"; DRAIN_S="${DRAIN_S:-120}"
V=/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2
P="$HERE/patches"

stop() {
  docker ps -q --filter "name=^${CONTAINER}\$" | grep -q . || { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; echo "not running"; return 0; }
  echo "draining $CONTAINER (up to ${DRAIN_S} s)"
  for _ in $(seq 1 "$DRAIN_S"); do
    n=$(curl -s -m 5 "http://127.0.0.1:${PORT}/metrics" | awk '/^vllm:num_requests_running/{s+=$2} END{print s+0}')
    [ "${n:-0}" = 0 ] && break
    sleep 1
  done
  docker stop -t 90 "$CONTAINER" >/dev/null && docker rm "$CONTAINER" >/dev/null && echo "stopped"
}

PROFILE="${1:-fastest}"
[ "$PROFILE" = stop ] && { stop; exit 0; }

# overlay file -> path inside the image
COMMON=(
  "draft/speculator.py:$V/speculator.py"
  "draft/dspark.py:$V/nvidia/dspark.py"
  "draft/gemv_q8.py:$V/nvidia/ops/gemv_q8.py"
  "draft/draft_vocab.json:/patches/dvocab/draft_vocab.json"
  "moe/moe_sm120.py:$V/nvidia/ops/moe_sm120.py"
)
INT8=(
  "int8/gemv.py:$V/nvidia/ops/gemv.py"
  "int8/model.py:$V/nvidia/model.py"
  "int8/w8.py:$V/nvidia/ops/w8.py"
)
ATTENTION=(
  "attention/attention_sm90.py:$V/nvidia/ops/attention_sm90.py"
  "attention/attention.py:$V/nvidia/attention.py"
)
LOOP=(
  "loop/ops_loop.py:$V/nvidia/ops/loop.py"
  "loop/loop.py:$V/nvidia/loop.py"
  "loop/sconv.py:$V/nvidia/sconv.py"
)
ENV=(INKLING_DRAFT_VOCAB=/patches/dvocab/draft_vocab.json INKLING_DRAFT_VOCAB_N=32768
     INKLING2_DRAFT_QUANT=int8 INKLING2_DRAFT_QUANT_LAYERS=1)
case "$PROFILE" in
  lossless) OVERLAYS=("${COMMON[@]}" "gemv/gemv.py:$V/nvidia/ops/gemv.py"
                      "attention-exact/attention_sm90.py:$V/nvidia/ops/attention_sm90.py"
                      "attention-exact/attention.py:$V/nvidia/attention.py") ;;
  fast)     OVERLAYS=("${COMMON[@]}" "${INT8[@]}" "${ATTENTION[@]}"); ENV+=(INKLING_W8=attn,head INKLING_W8_FMT=int8) ;;
  fastest)  OVERLAYS=("${COMMON[@]}" "${INT8[@]}" "${LOOP[@]}" "${ATTENTION[@]}"); ENV+=(INKLING_W8=attn,head INKLING_W8_FMT=int8) ;;
  *) echo "unknown profile '$PROFILE' (fastest | fast | lossless | stop)" >&2; exit 2 ;;
esac

(cd "$P" && sha256sum --quiet -c SHA256SUMS) || { echo "patch checksum mismatch: refusing to start" >&2; exit 1; }
CMD=(docker run -d --name "$CONTAINER" --gpus all --ipc=host -p "$BIND:$PORT:8000")
for e in "${ENV[@]}"; do CMD+=(-e "$e"); done
CMD+=(-v "$MODEL_DIR:/models/Inkling-2-Mini:ro" -v "$CACHE_DIR/root:/root/.cache" -v "$CACHE_DIR/triton:/root/.triton")
for o in "${OVERLAYS[@]}"; do CMD+=(-v "$P/${o%%:*}:${o#*:}:ro"); done
CMD+=("$IMAGE" /models/Inkling-2-Mini
  --served-model-name thinkingmachines/Inkling-2-Mini
  --tensor-parallel-size 1
  --max-model-len 262144
  --reasoning-parser inkling
  --enable-auto-tool-choice
  --tool-call-parser inkling
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
  --safetensors-load-strategy eager
  --max-num-batched-tokens 8192
  --moe-backend humming
  --kv-cache-memory-bytes "$KV_CACHE"
  --gpu-memory-utilization "$GPU_UTIL")

if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
[ -f "$MODEL_DIR/config.json" ] || { echo "no model in $MODEL_DIR (run ./download.sh, or set MODEL_DIR)" >&2; exit 1; }
mkdir -p "$CACHE_DIR/root" "$CACHE_DIR/triton"
docker ps -aq --filter "name=^${CONTAINER}\$" | grep -q . && { echo "container $CONTAINER exists: ./serve.sh stop first" >&2; exit 1; }
t0=$(date +%s)
"${CMD[@]}" >/dev/null
echo "starting profile '$PROFILE' as $CONTAINER on $BIND:$PORT"
until curl -sf -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]; then
    echo "the container exited during startup; last log lines:" >&2; docker logs --tail 30 "$CONTAINER" >&2; exit 1
  fi
  sleep 3
done
echo "ready after $(( $(date +%s) - t0 )) s: http://127.0.0.1:${PORT}/v1 (model thinkingmachines/Inkling-2-Mini)"
