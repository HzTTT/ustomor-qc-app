#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

command -v agent-browser >/dev/null 2>&1 || {
  echo "agent-browser 未安装或不在 PATH：请先安装 agent-browser" >&2
  exit 1
}

mkdir -p output/playwright

SMOKE_BASE_URL="${SMOKE_BASE_URL:-http://localhost:8000}"
SMOKE_USER="${SMOKE_USER:-Sean}"
SMOKE_PASS="${SMOKE_PASS:-0357zaqxswcde}"
export SMOKE_BASE_URL SMOKE_USER SMOKE_PASS

json_center() {
  python3 -c '
import json,sys,re
s=sys.stdin.read()
m=re.findall(r"\{[\s\S]*\}", s)
if not m:
  raise SystemExit("no json object found")
b=json.loads(m[-1])
cx=b["x"] + b["width"]/2
cy=b["y"] + b["height"]/2
print(f"{int(round(cx))} {int(round(cy))}")
'
}

drag_by_boxes() {
  local src_x="$1"
  local src_y="$2"
  local dst_x="$3"
  local dst_y="$4"
  ab mouse move "$src_x" "$src_y"
  ab mouse down left
  ab mouse move "$dst_x" "$dst_y"
  ab mouse up left
}

SESSION="smoke-touch-dnd-$(date +%s)"
ab() { agent-browser --session "$SESSION" "$@"; }

ab close >/dev/null 2>&1 || true

ab open "$SMOKE_BASE_URL/login?next=%2Freports%2Ftags%3Ftouch%3D1"
ab wait 'input[name="username"]'

ab fill 'input[name="username"]' "$SMOKE_USER"
ab fill 'input[name="password"]' "$SMOKE_PASS"
ab click 'form button'
ab wait --url '**/reports/tags*'

# 1) reports/tags: move a chip from available -> active and reorder within active.
ab open "$SMOKE_BASE_URL/reports/tags?touch=1"
ab wait '#row-fields'
ab wait '#available-fields [data-field="agent"]'

ab scrollintoview '#available-fields [data-field="agent"] [data-chip-handle]'
ab scrollintoview '#row-fields'

agent_box="$(ab get box '#available-fields [data-field="agent"] [data-chip-handle]')"
row_box="$(ab get box '#row-fields')"
if [[ "$agent_box" != *"{"* ]]; then
  echo "无法获取 agent handle box: $agent_box" >&2
  exit 1
fi
if [[ "$row_box" != *"{"* ]]; then
  echo "无法获取 row-fields box: $row_box" >&2
  exit 1
fi
read -r ax ay < <(printf '%s' "$agent_box" | json_center)
read -r rx ry < <(printf '%s' "$row_box" | json_center)
drag_by_boxes "$ax" "$ay" "$rx" "$ry"
ab wait '#row-fields [data-field="agent"]'

ab scrollintoview '#row-fields [data-field="agent"] [data-chip-handle]'
ab scrollintoview '#row-fields [data-field="platform"]'
agent_in_row_box="$(ab get box '#row-fields [data-field="agent"] [data-chip-handle]')"
platform_box="$(ab get box '#row-fields [data-field="platform"]')"
if [[ "$agent_in_row_box" != *"{"* ]]; then
  echo "无法获取 agent(in row) handle box: $agent_in_row_box" >&2
  exit 1
fi
if [[ "$platform_box" != *"{"* ]]; then
  echo "无法获取 platform box: $platform_box" >&2
  exit 1
fi
read -r a2x a2y < <(printf '%s' "$agent_in_row_box" | json_center)
plat_x="$(python3 -c '
import json,sys,re
s=sys.stdin.read()
m=re.findall(r"\{[\s\S]*\}", s)
if not m:
  raise SystemExit("no json object found")
b=json.loads(m[-1])
print(int(round(b["x"] + 6)))
' <<<"$platform_box")"
plat_y="$(python3 -c '
import json,sys,re
s=sys.stdin.read()
m=re.findall(r"\{[\s\S]*\}", s)
if not m:
  raise SystemExit("no json object found")
b=json.loads(m[-1])
print(int(round(b["y"] + b["height"]/2)))
' <<<"$platform_box")"
drag_by_boxes "$a2x" "$a2y" "$plat_x" "$plat_y"

# 2) settings/tags: pointer-based reorder in category list (best-effort; skip if <2 categories).
ab open "$SMOKE_BASE_URL/settings/tags?view=one&touch=1"
ab wait '#categoryList .draggable-item'

count="$(ab get count '#categoryList .draggable-item' | tr -dc '0-9')"
if [[ "${count:-0}" -ge 2 ]]; then
  ab scrollintoview '#categoryList .draggable-item:nth-child(2) [data-drag-handle]'
  second_handle_box="$(ab get box '#categoryList .draggable-item:nth-child(2) [data-drag-handle]')"
  first_item_box="$(ab get box '#categoryList .draggable-item:nth-child(1)')"
  if [[ "$second_handle_box" != *"{"* || "$first_item_box" != *"{"* ]]; then
    echo "无法获取 tags_manage box: second=$second_handle_box first=$first_item_box" >&2
  else
  read -r sx sy < <(printf '%s' "$second_handle_box" | json_center)
  fx="$(python3 -c '
import json,sys,re
s=sys.stdin.read()
m=re.findall(r"\{[\s\S]*\}", s)
if not m:
  raise SystemExit("no json object found")
b=json.loads(m[-1])
print(int(round(b["x"] + 12)))
' <<<"$first_item_box")"
  fy="$(python3 -c '
import json,sys,re
s=sys.stdin.read()
m=re.findall(r"\{[\s\S]*\}", s)
if not m:
  raise SystemExit("no json object found")
b=json.loads(m[-1])
print(int(round(b["y"] + 12)))
' <<<"$first_item_box")"
  drag_by_boxes "$sx" "$sy" "$fx" "$fy"
  fi
fi

ab screenshot --full output/playwright/smoke_touch_dnd.png
ab close

echo "✅ Smoke done: output/playwright/smoke_touch_dnd.png"
