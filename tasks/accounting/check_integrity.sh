#!/bin/bash
# Quick integrity check for accounting server files
# Run after any agent makes changes to verify nothing is corrupted

ERRORS=0
SERVER_DIR="$(dirname "$0")/server"

for f in "$SERVER_DIR"/*.py; do
    if ! python3 -c "compile(open('$f').read(), '$f', 'exec')" 2>/dev/null; then
        echo "CORRUPT: $f"
        ERRORS=$((ERRORS + 1))
        # Auto-restore from git
        git checkout -- "$f" 2>/dev/null && echo "  RESTORED from git"
    fi
done

# Check minimum file sizes (catch truncation)
for f in "$SERVER_DIR"/actions.py "$SERVER_DIR"/agent.py "$SERVER_DIR"/planner.py "$SERVER_DIR"/main.py; do
    lines=$(wc -l < "$f" 2>/dev/null || echo 0)
    if [ "$lines" -lt 50 ]; then
        echo "TRUNCATED: $f ($lines lines)"
        ERRORS=$((ERRORS + 1))
        git checkout -- "$f" 2>/dev/null && echo "  RESTORED from git"
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo "All server files OK"
else
    echo "$ERRORS file(s) had issues (auto-restored)"
fi
exit $ERRORS
