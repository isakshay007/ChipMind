# Pre-Launch Checklist (Before Sharing With Others)

**Run this before announcing or sharing the repo.** Ensures everything is configured so only **@isakshay007** controls `main`.

**First:** Read [OWNERSHIP_CHECKLIST.md](OWNERSHIP_CHECKLIST.md) to protect your authorship before anyone else.

---

## 1. Run Setup Script

```bash
make setup-github
```

This configures branch protection with **push restrictions** so only you can push to `main`.

---

## 2. Verify Branch Protection

Go to: **https://github.com/isakshay007/ChipMind/settings/branches**

- [ ] Rule exists for `main`
- [ ] **Restrict who can push to matching branches** is enabled
- [ ] Only **isakshay007** is in the allowed list
- [ ] Require status checks: `test` is listed
- [ ] Force pushes: Disabled
- [ ] Branch deletions: Disabled

---

## 3. Verify Default Branch

**Settings → General → Default branch**

- [ ] Default branch is `main` (or `master`)

---

## 4. Verify CODEOWNERS

`.github/CODEOWNERS` should contain:

```
* @isakshay007
```

- [ ] File exists and is correct

---

## 5. Test the Flow (Optional)

1. Create a second GitHub account or ask a friend to fork
2. They create a branch and open a PR
3. Confirm they **cannot** push directly to `main`
4. You merge the PR (only you can)

---

## 6. Repo Visibility

- [ ] **Public** — Anyone can see and fork. Only you can push to `main`.
- [ ] **Private** — Only you (and collaborators you add) can see. Use if not ready to share.

---

## 7. Optional: Collaborators

If you add collaborators under **Settings → Collaborators**:
- Grant **Read** or **Write** to non-main branches only
- They still cannot push to `main` (branch protection overrides)
- Or keep the repo without collaborators; contributors use forks + PRs

---

## Summary

| Item                         | Status |
|-----------------------------|--------|
| Branch protection           | Only @isakshay007 can push to main |
| Status checks               | `test` must pass |
| CODEOWNERS                  | @isakshay007 |
| CONTRIBUTING.md             | Explains fork → branch → PR workflow |
| PR template                 | Notes that only owner merges |

**Ready to share when all checks pass.**
