#!/usr/bin/env bash
# Benchmark runner — deterministic report, fixed iterations.
set -euo pipefail

cd "$(dirname "$0")"
REPORT="benchmark-results.txt"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  text-embedder benchmark report                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "Cores: $(nproc)"
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Write header to report
cat > "$REPORT" << EOF
text-embedder benchmark report
==============================

CPU:    $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
Cores:  $(nproc)
Date:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Iterations per benchmark: 5

EOF

run_bench() {
  local label="$1"
  local pattern="$2"
  echo "─── $label ───"
  echo "" >> "$REPORT"
  echo "$label" >> "$REPORT"
  echo "$(printf '%.0s─' $(seq 1 ${#label}))" >> "$REPORT"
  go test -bench="$pattern" -benchtime=5x -timeout 120s ./cmd/text-embedder/ 2>/dev/null \
    | grep -E '^(Benchmark|ok |---)' >> "$REPORT"
  # Re-run to show on stdout too
  go test -bench="$pattern" -benchtime=5x -timeout 120s ./cmd/text-embedder/ 2>/dev/null \
    | grep -E '(Benchmark|PASS|ok )' 
  echo ""
}

# ── 1. Sequential baseline ──────────────────────────────────────────────
echo "" >> "$REPORT"
echo "1. Sequential baseline (--workers=1)" >> "$REPORT"
echo "===================================" >> "$REPORT"
run_bench "1. Sequential baseline (--workers=1)" 'BenchmarkBatchHandler_workers_1$'

# ── 2. Worker scaling (batch=128) ───────────────────────────────────────
echo "" >> "$REPORT"
echo "2. Worker scaling — batch=128 texts" >> "$REPORT"
echo "===================================" >> "$REPORT"
run_bench "  workers=2"  'BenchmarkBatchHandler_workers_2$'
run_bench "  workers=4"  'BenchmarkBatchHandler_workers_4$'
run_bench "  workers=8"  'BenchmarkBatchHandler_workers_8$'

# ── 3. Provider batch sizes (workers=default) ───────────────────────────
echo "" >> "$REPORT"
echo "3. Batch sizes (workers=default)" >> "$REPORT"
echo "================================" >> "$REPORT"
run_bench "  single (1 text)"    'BenchmarkBatchHandler_single$'
run_bench "  small (16 texts)"   'BenchmarkBatchHandler_smallBatch$'
run_bench "  provider (128)"     'BenchmarkBatchHandler_providerBatchSize$'
run_bench "  max (256 texts)"    'BenchmarkBatchHandler_maxBatch$'

# ── 4. Concurrent callers ───────────────────────────────────────────────
echo "" >> "$REPORT"
echo "4. Concurrent callers (batch=128 each, workers=default)" >> "$REPORT"
echo "========================================================" >> "$REPORT"
run_bench "  1 caller"   'BenchmarkBatchHandler_concurrent/callers=1$'
run_bench "  2 callers"  'BenchmarkBatchHandler_concurrent/callers=2$'
run_bench "  4 callers"  'BenchmarkBatchHandler_concurrent/callers=4$'

# ── 5. JSON serialization ───────────────────────────────────────────────
echo "" >> "$REPORT"
echo "5. JSON serialization (batch response encode)" >> "$REPORT"
echo "==============================================" >> "$REPORT"
run_bench "  n=16"  'BenchmarkBatchJSON/n=16$'
run_bench "  n=128" 'BenchmarkBatchJSON/n=128$'
run_bench "  n=256" 'BenchmarkBatchJSON/n=256$'

echo "────────────────────────────────────────"
echo ""
echo "Report written to: $REPORT"
echo ""
cat "$REPORT"
