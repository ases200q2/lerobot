---
noteId: "5a223830ae8911f08159e78440657f13"
tags: []

---

# 🎯 Simple Sync Setup for LeRobot Fork

This setup is designed for **main branch only** synchronization with the upstream repository while keeping all other branches in your remote fork.

## 📁 Scripts

- `sync_main.sh` - **Main script** to sync only with upstream/main
- `cleanup_remote.sh` - **Cleanup script** to remove all branches except main from your fork
- `SIMPLE_SYNC_README.md` - This documentation

## 🚀 Quick Start

### 1. Clean Up Your Remote Repository (One Time)

First, remove all the unnecessary branches from your fork:

```bash
# See what would be deleted (dry run)
./cleanup_remote.sh --dry-run

# Actually delete all branches except main
./cleanup_remote.sh
```

### 2. Sync with Upstream

```bash
# Sync your main branch with upstream/main
./sync_main.sh
```

That's it! Simple and clean.

## 📋 What Each Script Does

### `sync_main.sh`
- ✅ Switches to main branch
- ✅ Fetches latest changes from upstream/main
- ✅ Merges upstream/main into your main branch
- ✅ Pushes updated main branch to your fork
- ✅ Automatically stashes/unstashes your changes

### `cleanup_remote.sh`
- ✅ Lists all branches that will be deleted
- ✅ Deletes all branches from your fork except main
- ✅ Shows progress and results
- ✅ Safe with confirmation prompt

## 🎯 Your Workflow

### Daily Sync
```bash
./sync_main.sh
```

### Before Starting New Work
```bash
# Make sure you're up-to-date
./sync_main.sh

# Create your feature branch
git checkout -b feature/my-feature
# ... work on your feature ...
git add .
git commit -m "Add my feature"
git push -u origin feature/my-feature
```

### Contributing Back
1. Push your feature branch to your fork
2. Go to https://github.com/ases200q2/lerobot
3. Create a Pull Request to `huggingface/lerobot`

## ✨ Benefits

- ✅ **Simple and focused** - only main branch sync
- ✅ **Clean repository** - no unnecessary branches
- ✅ **Easy to use** - single command sync
- ✅ **Safe operation** - automatic stashing
- ✅ **Fast execution** - no complex branch logic

## 🔧 Options

### sync_main.sh
- `--help` - Show help message
- `--force` - Force sync even with uncommitted changes

### cleanup_remote.sh
- `--help` - Show help message
- `--dry-run` - Show what would be deleted without deleting
- `--force` - Skip confirmation prompt

## ⚠️ Important Notes

- **Only main branch** will remain in your fork after cleanup
- **All other branches** will be permanently deleted from your fork
- **Your local branches** are not affected by cleanup
- **Upstream branches** are not affected by cleanup

## 🎉 Result

After running the cleanup, your fork will have:
- ✅ **Only main branch** (clean and simple)
- ✅ **Easy sync** with upstream/main only
- ✅ **No confusion** about which branch to sync
- ✅ **Fast and reliable** synchronization

This setup is perfect if you only want to work with the main branch and don't need the complexity of multiple branches.
