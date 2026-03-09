#!/usr/bin/env bash
# ChipMind GitHub Repository Setup
# Run after: gh auth login
# Usage: ./scripts/setup-github.sh  OR  make setup-github

set -e

REPO_OWNER="isakshay007"
REPO_NAME="ChipMind"
DESCRIPTION="AI-powered Verilog RTL design assistant with RAG, multi-agent orchestration, and compiler-in-the-loop"
TOPICS="verilog,rtl,llm,rag,langgraph,hardware-design,ai,python"
DEFAULT_BRANCH="main"

echo "=========================================="
echo "  ChipMind GitHub Setup"
echo "  Owner: Akshay Keerthi AS (@${REPO_OWNER})"
echo "=========================================="
echo ""

# Check gh CLI
if ! command -v gh &>/dev/null; then
    echo "ERROR: GitHub CLI (gh) is not installed."
    echo ""
    echo "Install it:"
    echo "  macOS:   brew install gh"
    echo "  Ubuntu:  sudo apt install gh"
    echo "  Windows: winget install GitHub.cli"
    echo ""
    echo "Then run: gh auth login"
    exit 1
fi

# Check auth
if ! gh auth status &>/dev/null; then
    echo "ERROR: Not logged in to GitHub. Run: gh auth login"
    exit 1
fi

echo "[1/6] Checking repository..."
if ! gh repo view "${REPO_OWNER}/${REPO_NAME}" &>/dev/null; then
    echo "      Creating repository ${REPO_OWNER}/${REPO_NAME}..."
    gh repo create "${REPO_OWNER}/${REPO_NAME}" --public --description="${DESCRIPTION}" 2>/dev/null || true
    if [ -d .git ]; then
        git remote add origin "https://github.com/${REPO_OWNER}/${REPO_NAME}.git" 2>/dev/null || git remote set-url origin "https://github.com/${REPO_OWNER}/${REPO_NAME}.git" 2>/dev/null || true
    fi
else
    echo "      Repository exists."
    if [ -d .git ]; then
        git remote add origin "https://github.com/${REPO_OWNER}/${REPO_NAME}.git" 2>/dev/null || git remote set-url origin "https://github.com/${REPO_OWNER}/${REPO_NAME}.git" 2>/dev/null || true
    fi
fi

echo "[2/6] Setting description..."
gh repo edit "${REPO_OWNER}/${REPO_NAME}" --description "${DESCRIPTION}" 2>/dev/null || true

echo "[3/6] Setting topics..."
gh repo edit "${REPO_OWNER}/${REPO_NAME}" --add-topic verilog --add-topic rtl --add-topic llm --add-topic rag --add-topic langgraph --add-topic hardware-design --add-topic ai --add-topic python 2>/dev/null || true

echo "[4/6] Enabling features..."
gh repo edit "${REPO_OWNER}/${REPO_NAME}" --enable-issues --enable-wiki=false 2>/dev/null || true

echo "[5/6] Setting default branch to ${DEFAULT_BRANCH}..."
# Default branch is set when first push - ensure main exists
if git rev-parse --verify ${DEFAULT_BRANCH} &>/dev/null; then
    git push -u origin ${DEFAULT_BRANCH} 2>/dev/null || true
fi

echo "[6/6] Configuring branch protection..."
# Branch protection - ONLY @isakshay007 can push to main. Others must use PRs.
# Requires at least one push first so Actions workflow exists
if gh api repos/${REPO_OWNER}/${REPO_NAME}/branches/${DEFAULT_BRANCH} &>/dev/null; then
    RESTRICTIONS='{"users":["isakshay007"],"teams":[]}'
    if gh api repos/${REPO_OWNER}/${REPO_NAME}/branches/${DEFAULT_BRANCH}/protection \
        -X PUT \
        -H "Accept: application/vnd.github+json" \
        -f required_status_checks='{"strict":true,"contexts":["test"]}' \
        -f enforce_admins=false \
        -f required_pull_request_reviews='{"required_approving_review_count":0,"dismiss_stale_reviews":false}' \
        -f restrictions="$RESTRICTIONS" \
        -F allow_force_pushes=false \
        -F allow_deletions=false 2>/dev/null; then
        echo "      Branch protection enabled. Only @${REPO_OWNER} can push to main."
    else
        echo "      Note: Enable manually in Settings → Branches: restrict push to @${REPO_OWNER} only."
    fi
else
    echo "      Push your code first ('git push -u origin main'), then run this script again for branch protection."
fi

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Push your code:  git push -u origin main"
echo "  2. Add secrets (optional): Settings → Secrets → GROQ_API_KEY, NVIDIA_API_KEY"
echo "  3. Verify: https://github.com/${REPO_OWNER}/${REPO_NAME}"
echo ""
