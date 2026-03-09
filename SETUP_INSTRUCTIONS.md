# ChipMind — One-Time Setup Instructions

Follow these steps to create and configure your GitHub repository.

**Access model:** Only **@isakshay007** (Akshay Keerthi AS) can push to or merge into `main`. Contributors fork, work on their branches, and submit PRs. The setup script enforces branch protection with push restrictions.

**Ownership protection:** Before sharing, see [.github/OWNERSHIP_CHECKLIST.md](.github/OWNERSHIP_CHECKLIST.md) to ensure no one can claim the project as theirs.

---

## 1. Install GitHub CLI

```bash
# macOS
brew install gh

# Or if you get permission errors, run first:
# sudo chown -R $(whoami) /opt/homebrew
```

---

## 2. Log in to GitHub

```bash
gh auth login
```

Choose: GitHub.com → HTTPS → Login with browser. Authenticate with your account **isakshay007**.

---

## 3. Initialize Git (if not done)

```bash
cd /Users/akshay/Desktop/ChipMind
git init
git add .
git commit -m "Initial commit: ChipMind RTL design assistant"
```

---

## 4. Run Automated Setup

```bash
make setup-github
```

This will:
- Create the repo `https://github.com/isakshay007/ChipMind` (if it doesn't exist)
- Set description and topics
- Configure branch protection (after first push)
- Enable Issues, disable Wikis

---

## 5. Push Your Code

```bash
git branch -M main
git remote add origin https://github.com/isakshay007/ChipMind.git
git push -u origin main
```

---

## 6. Run Setup Again (for Branch Protection)

After the first push, the `test` workflow will run. Then:

```bash
make setup-github
```

This enables branch protection requiring the `test` check to pass before merge.

---

## 7. Optional: Add API Secrets

In GitHub: **Settings → Secrets and variables → Actions → New repository secret**

- `GROQ_API_KEY` — for tests that need LLM
- `NVIDIA_API_KEY` — if using NVIDIA NIM

---

## 8. Final Checklist (Before Sharing)

See [.github/REPO_CHECKLIST.md](.github/REPO_CHECKLIST.md) to verify branch protection, CODEOWNERS, and that only you can push to `main`.

---

## Done!

Your repo is at: **https://github.com/isakshay007/ChipMind**

Only **@isakshay007** can push to or merge into `main`. Contributors work via forks and Pull Requests.
