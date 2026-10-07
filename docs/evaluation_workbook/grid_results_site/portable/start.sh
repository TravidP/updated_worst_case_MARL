#!/bin/sh
set -eu
TASK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$(uname -s)" in
  Darwin) PLATFORM=macos ;;
  Linux) PLATFORM=linux ;;
  *) echo 'Use Start-Windows.cmd on Windows.'; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64|amd64) ARCH=amd64 ;;
  arm64|aarch64) ARCH=arm64 ;;
  *) echo 'Unsupported CPU. Use: python3 server.py'; exit 1 ;;
esac
PROGRAM="$TASK_DIR/bin/$PLATFORM-$ARCH/cbwce-viewer"
if [ ! -x "$PROGRAM" ]; then chmod +x "$PROGRAM"; fi
exec "$PROGRAM" "$@"
