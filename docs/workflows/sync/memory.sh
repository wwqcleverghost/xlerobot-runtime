#!/bin/bash
# Sync the agent memory folders of the laptop and the lab computer, both ways. Run on the laptop (NUS VPN on).
# The lab computer's address changes: LAB_IP=10.249.x.x bash docs/workflows/sync/memory.sh
# ponytail: of two files with the same name the newer one wins whole (an older edit to the same file is lost);
# nothing is ever deleted, so a memory removed on one side comes back from the other.
set -e
HERE=$HOME/.claude/projects/-home-wwq-xlerobot-runtime/memory
THERE=marmotlab:.claude-wwq/projects/-home-marmotlab-wwq-lerobot/memory
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o HostKeyAlias=[localhost]:2223 ${LAB_IP:+-o HostName=$LAB_IP}"
T=$(mktemp)
rsync -e "$SSH" "$THERE/MEMORY.md" "$T"
rsync -e "$SSH" -tu --exclude MEMORY.md "$THERE"/*.md "$HERE"/
rsync -e "$SSH" -tu --exclude MEMORY.md "$HERE"/*.md "$THERE"/
# index: one line per file, the laptop's wording first
cat "$HERE/MEMORY.md" "$T" | awk 'match($0, /\]\([^)]*\)/) { k = substr($0, RSTART, RLENGTH); if (!(k in seen)) { seen[k]; print } }' > "$T.new"
cp "$T.new" "$HERE/MEMORY.md"
rsync -e "$SSH" "$T.new" "$THERE/MEMORY.md"
rm -f "$T" "$T.new"
echo "$(wc -l < "$HERE/MEMORY.md") memories on both machines"
