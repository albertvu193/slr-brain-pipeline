#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# SLR Brain Pipeline — One-Click Mac Launcher
#
# Double-click this file in Finder to:
#   1. Clone (or update) the repo
#   2. Set up a Python virtual environment & install dependencies
#   3. Start the Flask app
#   4. Auto-open Chrome to http://localhost:8080
#
# To install: save this file anywhere, then run once in Terminal:
#   chmod +x launch.command
# After that, just double-click it in Finder.
# ──────────────────────────────────────────────────────────────

set -e

REPO_URL="https://github.com/albertvu193/slr-brain-pipeline.git"
APP_DIR="$HOME/slr-brain-pipeline"
PORT=8080
URL="http://localhost:$PORT"

echo "================================================"
echo "  SLR Brain Pipeline — Launcher"
echo "================================================"
echo ""

# ── 1. Clone or pull latest ──────────────────────────────────
if [ -d "$APP_DIR/.git" ]; then
    echo "→ Repo found at $APP_DIR — pulling latest changes..."
    cd "$APP_DIR"
    git pull origin main || git pull origin master || echo "  (pull skipped — working with local copy)"
else
    echo "→ Cloning repo into $APP_DIR..."
    if [ -d "$APP_DIR" ]; then
        echo "  Removing old non-git directory..."
        rm -rf "$APP_DIR"
    fi
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi
echo ""

# ── 2. Check for Python 3 ───────────────────────────────────
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: Python 3 is not installed."
    echo "Install it from https://www.python.org/downloads/ or run:"
    echo "  brew install python"
    echo ""
    echo "Press any key to close..."
    read -n 1
    exit 1
fi
echo "→ Using Python: $($PY --version)"
echo ""

# ── 3. Create / reuse virtual environment ────────────────────
VENV_DIR="$APP_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment..."
    $PY -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "→ Virtual environment active"
echo ""

# ── 4. Install / update dependencies ────────────────────────
echo "→ Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "→ Dependencies ready"
echo ""

# ── 5. Kill any previous instance on the same port ──────────
if lsof -ti:$PORT &>/dev/null; then
    echo "→ Stopping previous instance on port $PORT..."
    kill $(lsof -ti:$PORT) 2>/dev/null || true
    sleep 1
fi

# ── 6. Start the app in the background ──────────────────────
echo "→ Starting SLR Brain on $URL ..."
$PY app.py &
APP_PID=$!
echo "  PID: $APP_PID"
echo ""

# ── 7. Wait for the server to be ready ──────────────────────
echo "→ Waiting for server..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w '' "$URL" 2>/dev/null; then
        echo "  Server is up!"
        break
    fi
    sleep 0.5
done
echo ""

# ── 8. Open in Chrome (fall back to default browser) ────────
echo "→ Opening in Chrome..."
if [ -d "/Applications/Google Chrome.app" ]; then
    open -a "Google Chrome" "$URL"
else
    echo "  Chrome not found — opening in default browser..."
    open "$URL"
fi
echo ""

# ── 9. Keep terminal open & handle Ctrl+C gracefully ────────
echo "================================================"
echo "  SLR Brain is running at $URL"
echo "  Press Ctrl+C to stop the server."
echo "================================================"
echo ""

cleanup() {
    echo ""
    echo "→ Shutting down server (PID $APP_PID)..."
    kill $APP_PID 2>/dev/null || true
    wait $APP_PID 2>/dev/null || true
    echo "→ Server stopped. You can close this window."
    exit 0
}
trap cleanup INT TERM

# Wait for the app process so Ctrl+C works
wait $APP_PID
