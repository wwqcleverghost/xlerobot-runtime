#!/bin/bash
# Where every machine stands, one line each: code commit, commits behind origin/main, uncommitted files, our training runs.
# Run on the laptop with the NUS VPN on:  bash docs/workflows/sync/where.sh
# The lab computer's address changes:     LAB_IP=10.249.x.x bash docs/workflows/sync/where.sh
look='for d in "$1" ~/wwq/lerobot /media/data/wwq/xlerobot-runtime ~/wwq/xlerobot-runtime; do [ -d "$d/.git" ] && break; done
cd "$d" 2>/dev/null || { echo "no code checkout"; exit; }
git fetch -q origin 2>/dev/null
echo "$(git log -1 --format="%h %cd" --date=format:"%m-%d %H:%M") | behind main $(git rev-list --count HEAD..origin/main) | uncommitted $(git status --porcelain | wc -l) | training: $(pgrep -af lerobot_train | grep -o "wwq/outputs/[^/ ]*" | sed "s#.*/##" | sort -u | tr "\n" " ")"'
row() { printf '%-8s ' "$1"; shift; "$@" <<<"$look" 2>&1 | tail -1; }
S="ssh -o BatchMode=yes -o ConnectTimeout=6"
row laptop bash -s -- "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
row lab $S -o HostKeyAlias='[localhost]:2223' ${LAB_IP:+-o HostName=$LAB_IP} marmotlab bash -s
for h in server1 server2 server4 server6; do row $h $S $h bash -s; done
