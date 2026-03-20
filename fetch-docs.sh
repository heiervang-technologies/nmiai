#!/usr/bin/env bash
# Fetch all NM i AI 2026 docs from app.ainm.no/docs
# Modes:
#   fetch-docs.sh              - one-shot fetch all pages
#   fetch-docs.sh --poll [MIN] - poll every MIN minutes (default 5), notify on changes
set -euo pipefail

BASE_URL="https://app.ainm.no/docs"
OUT_DIR="docs/official"
DELAY=2  # seconds between requests
POLL_INTERVAL="${2:-5}"  # minutes, default 5
MASTER_PANE="%7"

# Map: doc path -> local file -> team keyword
# Team keywords: object-detection, tripletex, astar-island, google-cloud, general
declare -A PAGES=(
  ["_root"]="getting-started.md|general"
  ["norgesgruppen-data/overview"]="object-detection/overview.md|object-detection"
  ["norgesgruppen-data/submission"]="object-detection/submission.md|object-detection"
  ["norgesgruppen-data/scoring"]="object-detection/scoring.md|object-detection"
  ["norgesgruppen-data/examples"]="object-detection/examples.md|object-detection"
  ["tripletex/overview"]="tripletex/overview.md|tripletex"
  ["tripletex/sandbox"]="tripletex/sandbox.md|tripletex"
  ["tripletex/endpoint"]="tripletex/endpoint.md|tripletex"
  ["tripletex/scoring"]="tripletex/scoring.md|tripletex"
  ["tripletex/examples"]="tripletex/examples.md|tripletex"
  ["astar-island/overview"]="astar-island/overview.md|astar-island"
  ["astar-island/mechanics"]="astar-island/mechanics.md|astar-island"
  ["astar-island/endpoint"]="astar-island/endpoint.md|astar-island"
  ["astar-island/scoring"]="astar-island/scoring.md|astar-island"
  ["astar-island/quickstart"]="astar-island/quickstart.md|astar-island"
  ["google-cloud/overview"]="google-cloud/overview.md|google-cloud"
  ["google-cloud/setup"]="google-cloud/setup.md|google-cloud"
  ["google-cloud/deploy"]="google-cloud/deploy.md|google-cloud"
  ["google-cloud/services"]="google-cloud/services.md|google-cloud"
)

EXTRA_PAGES=(
  "https://app.ainm.no/rules|rules.md|general"
)

cd "$(dirname "$0")"
mkdir -p "$OUT_DIR"/{object-detection,tripletex,astar-island,google-cloud}

# Python: extract <article> content from HTML, write to temp file for pandoc
EXTRACTOR=$(cat <<'PYEOF'
import sys, re
html = sys.stdin.read()
# Prefer <article> tag, fall back to role="main"
match = re.search(r'<article[^>]*>(.*?)</article>', html, re.DOTALL)
if not match:
    match = re.search(r'<main[^>]*>(.*?)</main>', html, re.DOTALL)
if match:
    print(match.group(1))
else:
    print(html)
PYEOF
)

# Fetch a single page, return the converted markdown on stdout
# Args: url
fetch_content() {
  local url="$1"
  local tmpfile
  tmpfile=$(mktemp)
  local http_code

  http_code=$(curl -s -w '%{http_code}' -o "$tmpfile" \
    -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
    -H "Accept: text/html,application/xhtml+xml" \
    -L "$url")

  if [[ "$http_code" -ge 200 && "$http_code" -lt 400 ]]; then
    local content
    local fragment
    fragment=$(mktemp)
    # Extract <article> or <main> content, then convert to GFM markdown via pandoc
    python3 -c "$EXTRACTOR" < "$tmpfile" > "$fragment" 2>/dev/null
    content=$(pandoc -f html -t gfm --wrap=none "$fragment" 2>/dev/null || cat "$fragment")
    rm -f "$tmpfile" "$fragment"
    echo "$content"
    return 0
  else
    rm -f "$tmpfile"
    return 1
  fi
}

# Fetch and write a page, returns "changed" or "unchanged" or "new" or "failed"
# Args: url outfile
fetch_page() {
  local url="$1"
  local outfile="$2"
  local outpath="$OUT_DIR/$outfile"

  local content
  if ! content=$(fetch_content "$url"); then
    echo "failed"
    return
  fi

  # Write to temp, compare with existing
  local tmpout
  tmpout=$(mktemp)
  printf '%s\n' "$content" > "$tmpout"

  if [[ ! -f "$outpath" ]]; then
    mv "$tmpout" "$outpath"
    echo "new"
  elif ! diff -q "$tmpout" "$outpath" &>/dev/null; then
    mv "$tmpout" "$outpath"
    echo "changed"
  else
    rm -f "$tmpout"
    echo "unchanged"
  fi
}

# Notify master orchestrator about doc changes for a specific team
# Args: team changed_files_description
notify_master() {
  local team="$1"
  local details="$2"
  local my_pane
  my_pane=$(tmux-tool current 2>/dev/null || echo "%?")

  local msg="<agent id=\"docs-watcher\" role=\"docs-poller\" pane=\"${my_pane}\">DOCS UPDATED: ${team} docs changed on app.ainm.no. Updated files: ${details}. Please route to the relevant team lead for review.</agent>"

  echo "[notify] Sending to master $MASTER_PANE: $team docs changed"
  tmux-tool send "$MASTER_PANE" "$msg" 2>/dev/null || true
  sleep 0.5
  tmux send-keys -t "$MASTER_PANE" Enter 2>/dev/null || true
}

# ── One-shot mode ──
do_fetch_all() {
  echo "=== NM i AI 2026 Docs Fetcher (one-shot) ==="
  echo "Output: $OUT_DIR | Delay: ${DELAY}s"
  echo ""

  local count=0
  local total=${#PAGES[@]}
  for path in "${!PAGES[@]}"; do
    count=$((count + 1))
    IFS='|' read -r outfile team <<< "${PAGES[$path]}"
    local url="$BASE_URL/$path"
    [[ "$path" == "_root" ]] && url="$BASE_URL"
    echo -n "[$count/$total] $url -> $outfile ... "
    local result
    result=$(fetch_page "$url" "$outfile")
    echo "$result"
    [[ $count -lt $total ]] && sleep "$DELAY"
  done

  for entry in "${EXTRA_PAGES[@]}"; do
    IFS='|' read -r url outfile team <<< "$entry"
    echo -n "[extra] $url -> $outfile ... "
    local result
    result=$(fetch_page "$url" "$outfile")
    echo "$result"
    sleep "$DELAY"
  done

  echo ""
  echo "=== Done ==="
}

# ── Poll mode ──
do_poll() {
  echo "=== NM i AI 2026 Docs Poller ==="
  echo "Checking every ${POLL_INTERVAL} minutes | Output: $OUT_DIR"
  echo "Master pane: $MASTER_PANE"
  echo ""

  while true; do
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] Polling for changes..."

    # Track changes per team
    declare -A team_changes=()

    for path in "${!PAGES[@]}"; do
      IFS='|' read -r outfile team <<< "${PAGES[$path]}"
      local url="$BASE_URL/$path"
      [[ "$path" == "_root" ]] && url="$BASE_URL"
      local result
      result=$(fetch_page "$url" "$outfile")

      if [[ "$result" == "changed" || "$result" == "new" ]]; then
        echo "  * $result: $outfile ($team)"
        team_changes[$team]+="$outfile "
      fi
      sleep "$DELAY"
    done

    for entry in "${EXTRA_PAGES[@]}"; do
      IFS='|' read -r url outfile team <<< "$entry"
      local result
      result=$(fetch_page "$url" "$outfile")

      if [[ "$result" == "changed" || "$result" == "new" ]]; then
        echo "  * $result: $outfile ($team)"
        team_changes[$team]+="$outfile "
      fi
      sleep "$DELAY"
    done

    # Notify master for each team with changes
    if [[ ${#team_changes[@]} -gt 0 ]]; then
      echo "[$timestamp] Changes detected!"
      for team in "${!team_changes[@]}"; do
        local files="${team_changes[$team]}"
        notify_master "$team" "$files"
      done
    else
      echo "[$timestamp] No changes."
    fi

    unset team_changes
    echo "[$timestamp] Next poll in ${POLL_INTERVAL} minutes."
    sleep "$((POLL_INTERVAL * 60))"
  done
}

# ── Main ──
case "${1:-}" in
  --poll)
    do_poll
    ;;
  *)
    do_fetch_all
    ;;
esac
