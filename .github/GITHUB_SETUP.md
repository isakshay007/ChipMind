# GitHub Repository Setup

## Quick setup (automated)

If you have the [GitHub CLI](https://cli.github.com/) installed:

```bash
# 1. Install gh (if needed)
brew install gh          # macOS
# or: sudo apt install gh   # Ubuntu

# 2. Log in to GitHub
gh auth login

# 3. Run setup
make setup-github
```

This configures: repo description, topics, branch protection, and PR defaults.

---

## Manual checklist

Use this if you prefer to configure settings in the GitHub web UI.

## 1. Repository Settings

**Settings → General**
- [ ] Add description: `AI-powered Verilog RTL design assistant with RAG, multi-agent orchestration, and compiler-in-the-loop`
- [ ] Add topics: `verilog`, `rtl`, `llm`, `rag`, `langgraph`, `hardware-design`, `ai`, `python`
- [ ] Enable "Issues" and "Discussions" (optional)
- [ ] Disable "Wikis" if not needed

## 2. Branch Protection (main / master)

**Settings → Branches → Add branch protection rule**

- [ ] Branch name pattern: `main` (or `master`)
- [ ] **Require a pull request before merging**
  - [ ] Require approvals: 0
  - [ ] Dismiss stale reviews: No
- [ ] **Require status checks to pass before merging**
  - [ ] Add status check: `test` (from Actions workflow)
  - [ ] Require branches to be up to date
- [ ] **Restrict who can push to matching branches** → **Add @isakshay007 only**
  - Only you can push to or merge into `main`. Contributors work on branches and submit PRs.
- [ ] Do not allow bypassing (optional; owner can bypass with `false`)
- [ ] Allow force pushes: **No**
- [ ] Allow deletions: **No**

## 3. Pull Request Defaults

**Settings → General → Pull Requests**

- [ ] Allow merge commits (or squash / rebase per preference)
- [ ] Enable "Always suggest updating the branch"
- [ ] PR template is auto-loaded from `.github/PULL_REQUEST_TEMPLATE.md`

## 4. Actions / Secrets

**Settings → Secrets and variables → Actions**

For CI tests that need API keys (optional; tests can skip if missing):

- [ ] Add `GROQ_API_KEY` (optional, for API-dependent tests)
- [ ] Add `NVIDIA_API_KEY` (optional)

## 5. Default Branch

**Settings → General**

- [ ] Default branch: `main` or `master` (match your workflow)

## 6. CODEOWNERS

`.github/CODEOWNERS` is already set — `@isakshay007` is the code owner. GitHub will auto-request your review on PRs if you enable "Require review from Code Owners" in branch protection.

## 7. Optional: Dependabot

**Settings → Code security and analysis**

- [ ] Enable Dependabot alerts
- [ ] Enable Dependabot security updates

Create `.github/dependabot.yml` (optional) to auto-open PRs for dependency updates.
