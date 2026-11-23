#!/usr/bin/env bash
# The README quickstart, checked: build, start, wait for the app, ask one question (ADR 0024).
# Needs a running Docker; uses whatever .env exists, so CI runs it with none (no LLM, keywords).
set -euo pipefail

BASE=${BASE:-http://localhost:8000}

fail() {
  echo "FAIL: $1" >&2
  docker compose ps -a >&2
  docker compose logs seed app >&2
  exit 1
}

docker compose up --build -d || fail "compose did not start"

curl -sf --retry 30 --retry-delay 2 --retry-all-errors "$BASE/health/ready" >/dev/null \
  || fail "app never became ready"

projects=$(curl -sf "$BASE/api/info" | jq -c '.projects') || fail "/api/info"
echo "projects: $projects"
echo "$projects" | jq -e 'index("shop") != null' >/dev/null || fail "shop is not loaded"

answer=$(curl -sf "$BASE/api/ask" -H 'content-type: application/json' \
  -d '{"node": "int_order_payments", "project": "shop"}') || fail "/api/ask"
downstream=$(echo "$answer" | jq -c '[.table[].display_name] | sort')
echo "int_order_payments downstream: $downstream"
expected='["dim_customers","fct_orders","mart_orders_by_region","mart_revenue_daily"]'
[ "$downstream" = "$expected" ] || fail "expected $expected"

echo "quickstart ok"
