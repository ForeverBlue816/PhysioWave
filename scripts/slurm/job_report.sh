#!/bin/bash
# ============================================================================
# Collect everything about a batch of jobs into one pasteable report.
#
#   bash scripts/slurm/job_report.sh                     # all of today's
#   bash scripts/slurm/job_report.sh 54340312 54340320   # specific ids
#   bash scripts/slurm/job_report.sh physiowave_hbn_dl   # by job name
#
# Exists because "check the output" is several commands -- squeue for what is
# still running, sacct for what already exited and WHY (a job killed for memory
# and a job that failed on its first line both just disappear from squeue), and
# then the tail of two files per job. Getting one of those and not the others is
# how an OUT_OF_MEMORY array got read as sixteen successful completions.
#
# LINES controls how much of each log is shown (default 25).
#
# For the other question a maintenance notice raises -- will a pending job
# start before the window, or is it held until after -- see
# scripts/slurm/cluster_status.sh.
# ============================================================================

set -uo pipefail

LINES="${LINES:-25}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

ids=()
name_filter=""
for a in "$@"; do
    if [[ "${a}" =~ ^[0-9]+$ ]]; then ids+=("${a}"); else name_filter="${a}"; fi
done

echo "############################################################"
echo "# QUEUE  ($(date '+%F %H:%M:%S'))"
echo "############################################################"
squeue --me -o "%.12i %.20j %.9P %.2t %.10M %.6D %R" 2>/dev/null || \
    echo "(squeue unavailable)"

echo ""
echo "############################################################"
echo "# ACCOUNTING -- includes jobs that already exited"
echo "############################################################"
# State AND ExitCode AND MaxRSS: a job that vanished from squeue may have
# completed, failed on line one, or been killed by the OOM killer, and only
# this tells them apart.
if [[ "${#ids[@]}" -gt 0 ]]; then
    sacct -j "$(IFS=,; echo "${ids[*]}")" \
        --format=JobID,JobName%22,State,ExitCode,Elapsed,MaxRSS -X 2>/dev/null
else
    sacct -S "$(date '+%Y-%m-%d')" \
        --format=JobID,JobName%22,State,ExitCode,Elapsed,MaxRSS -X 2>/dev/null
fi || echo "(sacct unavailable)"

echo ""
echo "############################################################"
echo "# LOGS  (last ${LINES} lines of each)"
echo "############################################################"
logs=()
if [[ "${#ids[@]}" -gt 0 ]]; then
    for i in "${ids[@]}"; do
        while IFS= read -r f; do logs+=("${f}"); done < <(
            find . -maxdepth 1 \( -name "*_${i}.out" -o -name "*_${i}.err" \
                -o -name "*_${i}_*.out" -o -name "*_${i}_*.err" \) 2>/dev/null)
    done
elif [[ -n "${name_filter}" ]]; then
    while IFS= read -r f; do logs+=("${f}"); done < <(
        find . -maxdepth 1 -name "${name_filter}*" \
            \( -name '*.out' -o -name '*.err' \) -mtime -1 2>/dev/null)
else
    while IFS= read -r f; do logs+=("${f}"); done < <(
        find . -maxdepth 1 \( -name '*.out' -o -name '*.err' \) \
            -mtime -1 2>/dev/null)
fi

if [[ "${#logs[@]}" -eq 0 ]]; then
    echo "(no matching log files in $(pwd))"
    echo ""
    echo "Logs land in the directory sbatch was run from. If that was not here,"
    echo "cd there and rerun this, or: ls -lt ~/*.out ~/*.err | head"
    exit 0
fi

# Newest last, so the most recent thing is at the bottom where it is read first.
for f in $(ls -rt "${logs[@]}" 2>/dev/null); do
    sz=$(wc -c < "${f}" | tr -d ' ')
    echo ""
    echo "=== ${f}  (${sz} bytes) ==="
    if [[ "${sz}" -eq 0 ]]; then
        echo "(empty -- the job has not written anything yet, or never started)"
    else
        tail -n "${LINES}" "${f}"
    fi
done
