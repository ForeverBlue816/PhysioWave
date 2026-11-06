#!/bin/bash
# ============================================================================
# Is the cluster about to go down, and will my job start before it does?
#
#   bash scripts/slurm/cluster_status.sh
#   RUN_DIR=$PW_CKPT_ROOT/pretrain_eeg_c1_moe bash scripts/slurm/cluster_status.sh
#
# job_report.sh answers "what happened to my jobs". This answers the other
# question, the one a maintenance notice raises: a pending job whose --time
# would run past the start of a maintenance reservation is not delayed by it,
# it is held until AFTER the window. Twelve hours of downtime plus a 24 h job
# that was going to start in one hour becomes a job that starts in thirty-six,
# and squeue says only "(Priority)" or "(ReqNodeNotAvail)" about it.
#
# So this prints, in order: the reservations, what they cost the partition, the
# jobs, and then the arithmetic -- how long until the window, how long each
# pending job asked for, and how long it could ask for instead and still run.
#
# The last part uses the run's OWN measured epoch time from
# metrics_epoch.jsonl, because a shorter walltime is only useful if a whole
# epoch still fits in it: this trainer writes latest.pth at epoch boundaries
# and nowhere else, so a job that ends mid-epoch makes no progress at all.
# ============================================================================

set -uo pipefail

PARTITION="${PARTITION:-boost_usr_prod}"
RUN_DIR="${RUN_DIR:-${PW_CKPT_ROOT:-}/pretrain_eeg_c1_moe}"
NOW_S="$(date +%s)"

have() { command -v "$1" >/dev/null 2>&1; }

if ! have squeue; then
    echo "ERROR: no squeue on PATH. This runs on a CINECA login node, not on" >&2
    echo "       a laptop -- the SLURM client tools are what it reads." >&2
    exit 1
fi

# Seconds from a SLURM duration: D-HH:MM:SS, HH:MM:SS, MM:SS, or UNLIMITED.
# Printed durations are not a fixed number of fields, and treating "1-00:00:00"
# as one hour is the error that makes a 24 h job look like it fits anywhere.
dur_to_s() {
    local t="$1" d=0
    case "${t}" in
        ""|UNLIMITED|N/A|INVALID) echo ""; return;;
    esac
    if [[ "${t}" == *-* ]]; then d="${t%%-*}"; t="${t#*-}"; fi
    local IFS=:; local -a p=(${t}); unset IFS
    case "${#p[@]}" in
        3) echo $(( d*86400 + 10#${p[0]}*3600 + 10#${p[1]}*60 + 10#${p[2]} ));;
        2) echo $(( d*86400 + 10#${p[0]}*60 + 10#${p[1]} ));;
        1) echo $(( d*86400 + 10#${p[0]} ));;
        *) echo "";;
    esac
}

# Epoch seconds from a SLURM timestamp, 2026-09-03T08:00:00.
#
# `date -d` is GNU-only. On a BSD date it fails, st_s comes back empty, and the
# script then reports "no upcoming reservation found" -- which is the one wrong
# answer that matters, because it reads as "nothing to worry about" on the day
# an outage is announced. python3 is on every login node this runs on, so the
# fallback is real rather than decorative, and a total failure to parse is
# reported instead of being rounded down to zero.
ts_to_s() {
    local t="$1" s=""
    case "${t}" in ""|"(null)"|N/A|Unknown|None) echo ""; return;; esac
    s=$(date -d "${t}" +%s 2>/dev/null)
    if [[ -z "${s}" ]] && have python3; then
        s=$(python3 -c 'import sys,datetime
print(int(datetime.datetime.fromisoformat(sys.argv[1]).timestamp()))' \
            "${t}" 2>/dev/null)
    fi
    echo "${s}"
}

human() {
    local s="$1"
    [[ -z "${s}" ]] && { echo "?"; return; }
    if (( s < 0 )); then echo "已过去 $(( -s / 3600 ))h"; return; fi
    printf '%dh %02dm' $(( s / 3600 )) $(( (s % 3600) / 60 ))
}

echo "############################################################"
echo "# RESERVATIONS / MAINTENANCE   ($(date '+%F %H:%M:%S %Z'))"
echo "############################################################"
res_raw="$(scontrol show reservation 2>/dev/null)"
maint_start_s=""
maint_name=""
seen_res=0
unparsed=0
if [[ -z "${res_raw}" || "${res_raw}" == *"No reservations"* ]]; then
    echo "  none declared."
    echo "  NOTE: an announced outage often has no reservation yet. Check the"
    echo "        node states below and CINECA's own notice as well."
else
    # One reservation per RECORD, and scontrol writes a record over several
    # lines separated by a blank line. `read` reads lines, not records, so the
    # fields of one reservation would arrive as four unrelated iterations with
    # StartTime in one of them and Flags in another. awk's paragraph mode
    # (RS="") splits on the blank lines and the gsub flattens each record onto
    # the single line `read` can actually consume.
    while IFS= read -r rec; do
        [[ -z "${rec}" ]] && continue
        name=$(sed -n 's/.*ReservationName=\([^ ]*\).*/\1/p' <<<"${rec}")
        st=$(sed -n 's/.*StartTime=\([^ ]*\).*/\1/p' <<<"${rec}")
        en=$(sed -n 's/.*EndTime=\([^ ]*\).*/\1/p' <<<"${rec}")
        du=$(sed -n 's/.*Duration=\([^ ]*\).*/\1/p' <<<"${rec}")
        nc=$(sed -n 's/.*NodeCnt=\([^ ]*\).*/\1/p' <<<"${rec}")
        fl=$(sed -n 's/.*Flags=\([^ ]*\).*/\1/p' <<<"${rec}")
        users=$(sed -n 's/.*Users=\([^ ]*\).*/\1/p' <<<"${rec}")
        st_s=$(ts_to_s "${st}")
        en_s=$(ts_to_s "${en}")
        seen_res=1
        echo "  ${name}   flags=${fl:-none}   nodes=${nc:-?}   users=${users:-all}"
        if [[ -n "${st_s}" ]]; then
            echo "      starts  ${st}   ($(human $(( st_s - NOW_S ))) 后)"
        else
            echo "      starts  ${st}   (无法解析这个时间戳)"
            unparsed=1
        fi
        echo "      ends    ${en}   (duration ${du:-?})"
        # The one that matters is the soonest one that has not started and is
        # not somebody else's private allocation.
        if [[ -n "${st_s}" ]] && (( st_s > NOW_S )); then
            if [[ -z "${maint_start_s}" ]] || (( st_s < maint_start_s )); then
                maint_start_s="${st_s}"; maint_name="${name}"
                # The string scontrol printed, not a reformat of the epoch:
                # `date -d @seconds` is GNU-only and the reformat is the same
                # portability bug as parsing was, for no gain.
                maint_end_str="${en}"
            fi
        fi
    done < <(awk 'BEGIN{RS=""} {gsub(/\n/," "); print}' <<<"${res_raw}")
fi

echo ""
echo "############################################################"
echo "# PARTITION ${PARTITION}"
echo "############################################################"
sinfo -p "${PARTITION}" -o "%.20P %.6a %.10l %.6D %.6t %N" 2>/dev/null | head -20
echo ""
echo "  nodes offline and why (empty = all healthy):"
sinfo -R -p "${PARTITION}" -o "%.19H %.30E %.10u %N" 2>/dev/null | head -15

echo ""
echo "############################################################"
echo "# YOUR JOBS"
echo "############################################################"
squeue --me -o "%.12i %.24j %.9P %.2t %.11l %.11M %.6D %.20S %R" 2>/dev/null

# ---------------------------------------------------------------------------
# What one epoch costs, BEFORE the verdict rather than after it.
#
# The verdict depends on this number. Printed the other way round, the fit
# section confidently says "resubmit with --time=01:53:00" and the epoch
# section two screens later says that job completes zero epochs -- two true
# statements that contradict each other as advice, and the reader acts on the
# first one they meet.
# ---------------------------------------------------------------------------
WORST_EPOCH_S=""
if [[ -f "${RUN_DIR}/metrics_epoch.jsonl" ]] && have python3; then
    echo ""
    echo "############################################################"
    echo "# WHAT ONE EPOCH COSTS  (${RUN_DIR})"
    echo "############################################################"
    _worst_f="$(mktemp)"
    python3 - "${RUN_DIR}/metrics_epoch.jsonl" "${_worst_f}" <<'PYEOF'
import json, sys

rows = []
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
# The last five, not all of them: this filesystem has already changed speed
# once mid-run by a factor of three, and an average over the whole history
# quotes a walltime the cluster no longer delivers.
recent = [r for r in rows if "train/epoch_seconds" in r][-5:]
if not recent:
    print("  metrics_epoch.jsonl has no epoch_seconds yet.")
    raise SystemExit

print(f"  {'epoch':>6} {'train':>10} {'val':>9} {'total':>10}")
tot = []
for r in recent:
    t = float(r.get("train/epoch_seconds", 0.0))
    v = float(r.get("val/val_seconds", 0.0))
    tot.append(t + v)
    print(f"  {r.get('epoch', '?'):>6} {t / 3600:>9.2f}h {v / 60:>8.1f}m "
          f"{(t + v) / 3600:>9.2f}h")

worst = max(tot)
print(f"\n  slowest of the last {len(tot)}: {worst / 3600:.2f} h per epoch")
print("  (the slowest, not the mean: a walltime sized on the mean loses the")
print("   epoch that runs long, and losing it costs that whole epoch)")
with open(sys.argv[2], "w") as f:
    f.write(str(int(worst)))
PYEOF
    WORST_EPOCH_S="$(cat "${_worst_f}" 2>/dev/null)"
    rm -f "${_worst_f}"
fi

echo ""
echo "############################################################"
echo "# 结论：现在该做什么"
echo "############################################################"
if [[ -z "${maint_start_s}" ]]; then
    if [[ "${unparsed}" -eq 1 ]]; then
        echo "  一个或多个 reservation 的时间戳没能解析 —— 结论不可信。"
        echo "  手动看：scontrol show reservation"
    elif [[ "${seen_res}" -eq 1 ]]; then
        echo "  有 reservation，但都已经开始或已过去 —— 没有待到来的窗口。"
        echo "  作业 pending 的原因看上面 REASON 那列，不是维护挡的。"
    else
        echo "  没有找到待到来的维护窗口 —— 没什么要绕开的。"
        echo "  作业 pending 的原因看上面 REASON 那列。"
    fi
else
    gap=$(( maint_start_s - NOW_S ))
    # A 5% cushion. The reservation's start is the moment SLURM stops
    # scheduling into it; a job ending at exactly that second is a job betting
    # on the queue being punctual.
    usable=$(( gap * 95 / 100 ))
    hhmm="$(printf '%02d:%02d:00' $(( usable / 3600 )) $(( (usable % 3600) / 60 )))"
    echo "  下一个窗口 '${maint_name}' 还有 $(human ${gap}) 开始"
    [[ -n "${maint_end_str:-}" ]] && echo "  维护结束 ${maint_end_str}"
    echo ""
    pending=0
    while read -r jid tl st; do
        [[ -z "${jid}" ]] && continue
        pending=1
        want="$(dur_to_s "${tl}")"
        if [[ -z "${want}" ]]; then
            echo "  作业 ${jid}: time limit '${tl}' 无法解析，跳过。"
        elif (( want <= gap )); then
            echo "  作业 ${jid} 要 ${tl}（$(human ${want})），空档有 $(human ${gap}) —— 放得下。"
            echo "      不用动。还在 pending 是排队，不是维护挡的。"
        elif [[ -n "${WORST_EPOCH_S}" ]] && (( usable < WORST_EPOCH_S )); then
            # The case that makes this script worth having. Shrinking the
            # walltime looks like the obvious move and is the wrong one: this
            # trainer writes latest.pth at epoch boundaries and nowhere else,
            # so a job that ends mid-epoch burns the allocation and advances
            # the run by zero.
            echo "  作业 ${jid} 要 ${tl}，空档只有 $(human ${gap})。"
            echo "      **别缩 walltime。** 空档（$(human ${usable})）装不下一个 epoch"
            echo "      （最慢一轮 $(( WORST_EPOCH_S / 60 )) 分钟），而 latest.pth 只在 epoch"
            echo "      边界写 —— 这种作业跑满也存不下任何东西，纯烧机时。"
            echo "      -> 什么都别做，让它排在维护后面。"
        else
            echo "  作业 ${jid} 要 ${tl}（$(human ${want})），空档只有 $(human ${gap})。"
            echo "      -> SLURM 不会在窗口前起它，会一直等到维护结束之后。"
            if [[ -n "${WORST_EPOCH_S}" ]]; then
                echo "      -> 想用掉这个空档：--time=${hhmm}，约跑完 $(( usable / WORST_EPOCH_S )) 个 epoch。"
            else
                echo "      -> 想用掉这个空档：--time=${hhmm}。"
            fi
            # Lowering a pending job's limit in place, NOT scancel+sbatch.
            # SLURM lets a user decrease their own job's TimeLimit, and doing
            # it in place keeps the job id, the submit time and the priority
            # the job has been accruing since it queued. Cancelling and
            # resubmitting throws all three away and puts the job at the back
            # of a queue it had been climbing.
            echo "           scontrol update JobId=${jid} TimeLimit=${hhmm}"
            echo "         （改完再看一次 squeue --me，REASON 应该不再是 Reserved"
            echo "           for maintenance；仍然不行才用 scancel + sbatch 重提）"
        fi
    done < <(squeue --me -h -t PD -o "%i %l %S" 2>/dev/null)
    [[ "${pending}" -eq 0 ]] && echo "  没有 pending 的作业要检查。"
fi

echo ""
echo "############################################################"
echo "# 其他"
echo "############################################################"
echo "  账户余额:            saldo -b"
echo "  某个作业为何没起:    scontrol show job <jobid> | grep -E 'Reason|StartTime'"
echo "  已结束的作业:        bash scripts/slurm/job_report.sh"
