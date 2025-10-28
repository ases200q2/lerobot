---
noteId: "6c5c29d0ae8811f08159e78440657f13"
tags: []

---

# 🔄 Sync Scripts for LeRobot Fork

This directory contains scripts to help you stay up-to-date with the original `huggingface/lerobot` repository.

## 📁 Files

- `sync_upstream.sh` - Main sync script with full functionality
- `update.sh` - Simple shortcut script
- `SYNC_README.md` - This documentation

## 🚀 Quick Start

### Basic Usage

```bash
# Sync current branch with upstream
./sync_upstream.sh

# Or use the shortcut
./update.sh
```

### Sync All Branches

```bash
# Sync both main and dev branches
./sync_upstream.sh --all
```

### Sync Specific Branches

```bash
# Sync only main branch
./sync_upstream.sh --main

# Sync only dev branch (syncs with upstream/main)
./sync_upstream.sh --dev

# Sync specific branches
./sync_upstream.sh main dev
```

## 📋 Full Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-a, --all` | Sync all branches (main, dev) |
| `-m, --main` | Sync only main branch |
| `-d, --dev` | Sync only dev branch |
| `-f, --force` | Force sync even with uncommitted changes |
| `-q, --quiet` | Quiet mode (less output) |

## 🔧 What the Script Does

1. **Checks** that you're in a git repository
2. **Verifies** that `upstream` and `origin` remotes are configured
3. **Stashes** any uncommitted changes (unless `--force` is used)
4. **Fetches** latest changes from upstream
5. **Merges** upstream changes into your branches
6. **Pushes** updated branches to your fork
7. **Restores** any stashed changes

## 📋 Branch Mapping

- **`main` branch** → syncs with `upstream/main`
- **`dev` branch** → syncs with `upstream/main` (since upstream doesn't have dev)
- **Other branches** → sync with their corresponding upstream branch

## 🎯 Examples

### Daily Workflow

```bash
# Stay up-to-date with upstream
./update.sh --all

# Work on your feature
git checkout -b feature/my-feature
# ... make changes ...
git add .
git commit -m "Add my feature"
git push -u origin feature/my-feature
```

### Before Starting New Work

```bash
# Make sure you're up-to-date
./update.sh --main
git checkout -b feature/new-feature
```

### Force Sync (with uncommitted changes)

```bash
# Sync even if you have uncommitted changes
./update.sh --force
```

## ⚠️ Important Notes

- **Always commit or stash** your changes before syncing (unless using `--force`)
- The script will **automatically stash** uncommitted changes if needed
- **Conflicts** will need to be resolved manually if they occur
- Your **private fork** will be updated automatically

## 🔍 Troubleshooting

### "Upstream remote not found"
Make sure you have the upstream remote configured:
```bash
git remote -v
# Should show:
# origin    https://github.com/ases200q2/lerobot.git (fetch)
# origin    https://github.com/ases200q2/lerobot.git (push)
# upstream  https://github.com/huggingface/lerobot.git (fetch)
# upstream  https://github.com/huggingface/lerobot.git (push)
```

### "Not in a git repository"
Make sure you're in the root directory of your lerobot repository.

### Merge conflicts
If conflicts occur, resolve them manually:
```bash
# Edit conflicted files
git add .
git commit -m "Resolve merge conflicts"
git push origin <branch-name>
```

## 🎉 Benefits

- ✅ **Automated syncing** with upstream repository
- ✅ **Safe operation** with automatic stashing
- ✅ **Flexible options** for different use cases
- ✅ **Clear feedback** with colored output
- ✅ **Error handling** with helpful messages

## 🔗 Related

- [GitHub Fork Workflow](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
- [LeRobot Repository](https://github.com/huggingface/lerobot)
- [Your Fork](https://github.com/ases200q2/lerobot)
