# Publishing to GitHub — AI-in-Drug-Discovery

This file gives step-by-step instructions to create a GitHub repo and push the current project.

Important:
- This repo contains raw data and possibly model files; GitHub has limits for very large files.
- If you need to include large files (> 100MB), use Git LFS. Otherwise keep `/data/` and `/models/` in `.gitignore` or remove them before pushing.

Quick Steps

1) Make the publish script executable:
```bash
chmod +x publish_to_github.sh
```

2) Authenticate with GitHub CLI (choose the account you want to publish to):
```bash
# If you don't have 'gh' installed, follow: https://cli.github.com/
gh auth login
```

3) Run the publish script (example: public repo named `AI-in-Drug-Discovery`):
```bash
./publish_to_github.sh "AI-in-Drug-Discovery" public
```

What the script does
- Initializes git if not already
- Stages & commits all files (respecting `.gitignore` by default)
- Creates a new GitHub repository under your account using `gh`
- Pushes the initial commit to `origin/main`

If a repo already exists with the same name, the script will ask if you want to overwrite the remote configuration.

Optional: Use Git LFS for large files

```bash
# Track model files
git lfs install
git lfs track "models/*.pkl"
git add .gitattributes
# Commit & push the LFS-tracked files
git add models/best_model.pkl
git commit -m "Add model via Git LFS"
git push origin main
```

If you'd like a trimmed (lightweight) branch that excludes `/data/` and `/models/` (more appropriate for public GitHub sharing), create this branch and push it instead:

```bash
git checkout -b lightweight
# Remove large files from index while keeping them locally
git rm -r --cached data
git rm -r --cached models
git commit -m "Create lightweight branch - exclude raw data and models"
# Push the lightweight branch to GitHub main
git push origin lightweight:main -u
```

If you’d like, I can prepare a `lightweight` branch for you now (remove raw data/models from the git index) and you can push it with the script. Or I can help run the script commands if you want to do them step-by-step on your system.