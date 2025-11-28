#!/usr/bin/env bash

# publish_to_github.sh
# Create a repository on your GitHub account and push the current directory
# Usage: ./publish_to_github.sh [REPO_NAME] [public|private]

set -euo pipefail

REPO_NAME=${1:-"AI-in-Drug-Discovery"}
VISIBILITY=${2:-"public"}

# Make sure we're at repo root
ROOT_DIR=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -n "$ROOT_DIR" ] && [ "$(pwd)" != "$ROOT_DIR" ]; then
  echo "Switching to git root: $ROOT_DIR"
  cd "$ROOT_DIR"
fi

# Ensure git is installed
if ! command -v git >/dev/null 2>&1; then
  echo "git is required. Install git and re-run the script." >&2
  exit 1
fi

# Ensure gh is installed
if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. Install it from https://cli.github.com/" >&2
  exit 1
fi

# Initialize git if necessary
if [ ! -d ".git" ]; then
  echo "Initializing local git repository..."
  git init
fi

# Add files respecting .gitignore
echo "Adding files and committing..."

git add -A

# Commit with safe message
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  git commit -m "Update project files" || true
else
  git commit -m "Initial commit" || true
fi

# Determine visibility flag for gh create
VIS_FLAG="--public"
if [ "$VISIBILITY" == "private" ]; then
  VIS_FLAG="--private"
fi

# If remote origin exists, ask whether to reuse
if git remote get-url origin >/dev/null 2>&1; then
  echo "A remote 'origin' already exists: $(git remote get-url origin)"
  echo "To push to a new repo on your account, either remove the remote or choose to reuse this origin."
  read -p "Overwrite remote and create a new GitHub repo? (y/N): " -r
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    git remote remove origin
  else
    echo "Pushing to the existing remote origin..."
    git push origin HEAD:main -u
    echo "Done."
    exit 0
  fi
fi

# Create repository and push
echo "Creating GitHub repo: $REPO_NAME ($VISIBILITY)"

# Use gh to create repo and push
if gh repo create "$REPO_NAME" $VIS_FLAG --source=. --remote=origin --push; then
  echo "Repository created and pushed successfully."
  echo "Repo URL: https://github.com/$(gh api user --silent | jq -r .login)/$REPO_NAME"
else
  echo "gh repo create failed. If the repo exists, set the remote manually and push:\n  git remote add origin <URL> && git push -u origin main" >&2
fi

exit 0
