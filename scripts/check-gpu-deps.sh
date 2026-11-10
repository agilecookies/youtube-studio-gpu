#!/usr/bin/env bash
# Check that every URL referenced in GPU container deps returns HTTP 200.
# Run from any directory; paths are resolved relative to this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== GPU deps URL check ==="

# LTX-2 archive URL pinned in ltx/vast-onstart.sh (must return 200)
LTX2_SHA="a95ab856bf29407b6b066ede0abe1846050db56c"
LTX2_URL="https://github.com/Lightricks/LTX-2/archive/${LTX2_SHA}.tar.gz"

URLS=("${LTX2_URL}")

# Scan both requirements files for any explicit HTTP(S) package URLs
for f in "${REPO_DIR}/ltx/requirements.txt" "${REPO_DIR}/flux/requirements.txt"; do
  if [ -f "${f}" ]; then
    while IFS= read -r line; do
      # Strip comments and whitespace
      clean="${line%%#*}"
      clean="${clean// /}"
      if [[ "${clean}" =~ ^https?:// ]]; then
        URLS+=("${clean}")
      fi
    done < "${f}"
  fi
done

PASS=0
FAIL=0
for url in "${URLS[@]}"; do
  STATUS=$(curl -sSo /dev/null -w '%{http_code}' -L --max-time 20 "${url}" 2>/dev/null || echo "000")
  if [ "${STATUS}" = "200" ]; then
    printf "  PASS  %s\n" "${url}"
    PASS=$((PASS + 1))
  else
    printf "  FAIL  %s  (HTTP %s)\n" "${url}" "${STATUS}"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "${PASS} passed, ${FAIL} failed"
[ "${FAIL}" -eq 0 ] || exit 1
