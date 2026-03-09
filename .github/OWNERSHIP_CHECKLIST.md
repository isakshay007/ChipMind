# Ownership Protection Checklist

**Complete this before sharing ChipMind.** Ensures your authorship is legally and visibly established so no one can claim it as theirs.

---

## ✅ Already Done (in this repo)

| Item | Location | Status |
|------|----------|--------|
| LICENSE (MIT) | `LICENSE` | Copyright (c) 2025 Akshay Keerthi AS |
| Project authors | `pyproject.toml` | Akshay Keerthi AS |
| Code owner | `.github/CODEOWNERS` | @isakshay007 |
| README author | `README.md` | Linked to isakshay007 |
| Branch protection | `scripts/setup-github.sh` | Only @isakshay007 can push to main |
| AUTHORS file | `AUTHORS` | Lists you as creator |
| Package copyright | `chipmind/__init__.py` | Copyright header |

---

## 🔴 YOU MUST DO (Before Anyone Else)

### 1. Set Git Author (Before First Commit)

```bash
git config user.name "Akshay Keerthi AS"
git config user.email "YOUR_EMAIL@example.com"  # Use the email tied to your GitHub account
```

Your name and email are baked into every commit. Use the email from your GitHub account (https://github.com/settings/emails) so commits link to your profile.

### 2. Create the Repo and Push First

```bash
make setup-github
git add .
git commit -m "Initial commit: ChipMind - AI-powered RTL design assistant"
git push -u origin main
```

**Why:** The first push establishes you as the original author on GitHub. The commit graph and "Created by" attribution are permanent.

### 3. Verify GitHub Repo Owner

- Go to https://github.com/isakshay007/ChipMind
- Confirm the repo is under **your** account (isakshay007)
- The "Created" date and your profile are visible
- No one else can "take over" unless they compromise your account

### 4. Optional: Reserve PyPI Package Name

If you might publish to PyPI later:

```bash
# Create a minimal release or placeholder
pip install twine
# Publish 0.1.0 to PyPI (claim the "chipmind" name)
```

Otherwise someone could register `chipmind` on PyPI before you. You can do this later; not urgent for a portfolio project.

---

## What Protects You

| Protection | What it does |
|------------|--------------|
| **LICENSE** | Legal document. "Copyright (c) 2025 Akshay Keerthi AS" = you own it. MIT lets others use it but they must keep your notice. |
| **Git history** | Every commit has your name/email. Visible forever on GitHub. |
| **GitHub ownership** | Repo is under your account. You control it. Forks show "forked from isakshay007/ChipMind". |
| **Branch protection** | Only you can push to main. No one can overwrite your code. |
| **AUTHORS + __init__** | Copyright appears in the code itself. Copied code still carries your name. |

---

## What Cannot Be Fully Prevented

- **Forks:** Anyone can fork. GitHub shows "forked from isakshay007/ChipMind" so attribution exists.
- **Someone claiming credit elsewhere:** (e.g., resume, blog) No technical lock. Your LICENSE, git history, and published repo are evidence if needed.
- **Copy-paste without attribution:** MIT requires keeping the copyright notice. If they remove it, they violate the license.

---

## Final Verification

Before sharing publicly:

1. [ ] Git config: `git config user.name` and `git config user.email` are correct
2. [ ] LICENSE exists with your name
3. [ ] First commit is pushed to GitHub under your account
4. [ ] Repo URL: https://github.com/isakshay007/ChipMind
5. [ ] Run `make setup-github` so branch protection is active

**You're protected.** Push first, then share.
