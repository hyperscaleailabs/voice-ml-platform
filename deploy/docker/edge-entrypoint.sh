#!/bin/sh
# Verify the mounted bundle, then start the edge runtime.
# A failed verification exits non-zero so the supervisor (systemd / kubelet)
# restarts the previous bundle instead of running an unverified one.
set -eu
BUNDLE="${VMP_EDGE_BUNDLE:-/var/lib/vmp/edge/bundle}"
echo "edge: verifying bundle at ${BUNDLE}"
vmp edge bundle verify "${BUNDLE}"
echo "edge: bundle verified, starting runtime"
exec vmp serve api \
  --host "${VMP_EDGE_HOST:-127.0.0.1}" \
  --port "${VMP_EDGE_PORT:-8080}" \
  --backend "${VMP_EDGE_BACKEND:-echo}" \
  "$@"
