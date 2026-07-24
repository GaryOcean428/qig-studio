#!/usr/bin/env bash
# monitor_cradle_run2.sh — heartbeat + red-flag scan for the run-2 Modal cradle.
# Exit 0 = healthy or progressive; exit 2 = RED (needs immediate action); exit 1 = yellow/unknown.
set -euo pipefail

APP_ID="${CRADLE_APP_ID:-ap-1KcshQ00uYfU23RXbCQMCO}"
VOL="${CRADLE_VOL:-qig-cradle-vol}"
CKPT_DIR="${CRADLE_CKPT:-checkpoints/cradle_run2_20260724}"
STATE_DIR="${CRADLE_STATE_DIR:-/tmp/cradle-run2-monitor}"
mkdir -p "$STATE_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$STATE_DIR/check-$STAMP.log"
TRAJ="$STATE_DIR/basin_trajectory.jsonl"
PREV="$STATE_DIR/prev_metrics.json"
OUT="$STATE_DIR/latest_status.json"

exec > >(tee -a "$LOG") 2>&1
echo "=== cradle monitor $STAMP app=$APP_ID ==="

# 1) App still alive?
APP_JSON=$(uvx modal app list --json 2>/dev/null || echo '[]')
python3 - "$APP_ID" "$APP_JSON" <<'PY' || true
import json,sys
app_id=sys.argv[1]
try:
    apps=json.loads(sys.argv[2])
except Exception:
    apps=[]
hit=None
for a in apps:
    aid=a.get("app_id") or a.get("App ID") or ""
    if aid==app_id:
        hit=a; break
if not hit:
    print(f"RED: app {app_id} NOT in app list")
    sys.exit(2)
state=(hit.get("state") or hit.get("State") or "")
tasks=str(hit.get("tasks") or hit.get("Tasks") or "0")
print(f"app_state={state} tasks={tasks}")
if "stopped" in state.lower() or tasks in ("0","None"):
    print(f"RED: app not running (state={state} tasks={tasks})")
    sys.exit(2)
PY
APP_RC=$?

# 2) Pull trajectory heartbeat (volume commit lag possible)
if ! uvx modal volume get "$VOL" "$CKPT_DIR/basin_trajectory.jsonl" "$TRAJ" --force >/dev/null 2>&1; then
  echo "YELLOW: could not download basin_trajectory.jsonl (volume lag or path)"
  TRAJ_OK=0
else
  TRAJ_OK=1
fi

# 3) Scan recent app logs for red flags (bounded)
LOG_SNAP="$STATE_DIR/logs-$STAMP.txt"
timeout 40 uvx modal app logs "$APP_ID" >"$LOG_SNAP" 2>&1 || true

python3 - "$TRAJ" "$PREV" "$OUT" "$LOG_SNAP" "$APP_ID" "$STAMP" <<'PY'
import json, re, sys, time
from pathlib import Path

traj_path, prev_path, out_path, log_path, app_id, stamp = sys.argv[1:7]
red = []
yellow = []
info = {"ts": stamp, "app_id": app_id}

# --- trajectory (use LAST contiguous segment after a --fresh step-reset) ---
steps = []
phases = {}
if Path(traj_path).exists() and Path(traj_path).stat().st_size > 0:
    for line in Path(traj_path).read_text().splitlines():
        line=line.strip()
        if not line: continue
        try:
            d=json.loads(line)
        except Exception:
            continue
        s=d.get("step")
        if isinstance(s, int):
            steps.append(s)
            ph=d.get("phase") or "?"
            phases[ph]=phases.get(ph,0)+1
    # last segment: after the final step decrease (fresh relaunch appends without truncate)
    seg = steps
    for i in range(1, len(steps)):
        if steps[i] < steps[i-1]:
            seg = steps[i:]
    info["traj_n_all"]=len(steps)
    info["traj_n"]=len(seg)
    info["traj_max_step"]=max(seg) if seg else None
    info["traj_last_step"]=seg[-1] if seg else None
    info["traj_phases"]=phases
    info["traj_last_steps"]=seg[-5:] if seg else []
else:
    yellow.append("no_trajectory_yet")
    info["traj_n"]=0
    info["traj_max_step"]=None
    info["traj_last_step"]=None

prev = {}
if Path(prev_path).exists():
    try:
        prev=json.loads(Path(prev_path).read_text())
    except Exception:
        prev={}

now_last = info.get("traj_last_step") or 0
prev_last = prev.get("traj_last_step") or 0
prev_ts = prev.get("unix") or 0
unix = time.time()
info["unix"]=unix
info["step_delta_since_prev"]=now_last - prev_last
info["seconds_since_prev"]=round(unix - prev_ts, 1) if prev_ts else None

# Stall: last-segment step not advancing
if prev_ts and (unix - prev_ts) >= 180 and now_last <= prev_last and now_last > 0:
    red.append(f"STALL: traj last_step stuck at {now_last} for {unix-prev_ts:.0f}s")
elif prev_ts and (unix - prev_ts) >= 300 and now_last == 0:
    red.append("STALL: no trajectory steps after 5min")

# --- log scan ---
text = Path(log_path).read_text(errors="replace") if Path(log_path).exists() else ""
info["log_bytes"]=len(text)

def count(pat):
    return len(re.findall(pat, text, flags=re.I|re.M))

info["counts"]={
    "P1_warn": count(r"Pillar 1 \(Fluctuations\) violated"),
    "P3_CRITICAL": count(r"CRITICAL identity drift"),
    "Traceback": count(r"^Traceback"),
    "CoachUnreachable|LIVENESS FAILURE": count(r"LIVENESS FAILURE|CoachUnreachable|coach preflight FAILED"),
    "OOM|CUDA out of memory": count(r"out of memory|CUDA error|CUBLAS_STATUS"),
    "genesis_progress": count(r"\[joint\]\s+genesis\s+\d+/"),
    "gate_passed": count(r"LAUNCH CHECKLIST PASSED"),
}

# P1 spam threshold: more than ~1 per 40 steps is suspicious once we have steps
p1 = info["counts"]["P1_warn"]
if p1 >= 20:
    red.append(f"P1_SPAM: {p1} Pillar-1 warnings in log snapshot (always-firing channel back?)")
elif p1 >= 5:
    yellow.append(f"P1_elevated: {p1} warnings (rate-limit should keep these sparse)")

# One birth CRITICAL on a known-buggy pin is RED for instrument; count>1 is spam/regression.
p3c = info["counts"]["P3_CRITICAL"]
if p3c >= 3:
    red.append(f"P3_CRITICAL spam ({p3c}) — instrument broken or dissolution cascade")
elif p3c == 1:
    yellow.append("P3_CRITICAL once in log snapshot (known residual EMA-floor birth tick on 2.15.1; watch for more)")
elif p3c == 2:
    yellow.append("P3_CRITICAL twice — elevating watch; third becomes RED")

if info["counts"]["Traceback"] >= 1:
    red.append("Traceback in logs")
if info["counts"]["CoachUnreachable|LIVENESS FAILURE"] >= 1:
    red.append("COACH_LIVENESS failure / pause")
if info["counts"]["OOM|CUDA out of memory"] >= 1:
    red.append("OOM/CUDA error")

# progress line presence once past step 50
if now_last >= 50 and info["counts"]["genesis_progress"] == 0:
    yellow.append("past_step_50_but_no_genesis_progress_print_in_log_snapshot (log CLI lag possible)")

info["red"]=red
info["yellow"]=yellow
info["verdict"]="RED" if red else ("YELLOW" if yellow else "GREEN")

Path(out_path).write_text(json.dumps(info, indent=2))
Path(prev_path).write_text(json.dumps(info, indent=2))

print(json.dumps(info, indent=2))
print(f"VERDICT={info['verdict']}")
if red:
    sys.exit(2)
if yellow:
    sys.exit(1)
sys.exit(0)
PY
RC=$?
echo "monitor_exit=$RC"
exit $RC
