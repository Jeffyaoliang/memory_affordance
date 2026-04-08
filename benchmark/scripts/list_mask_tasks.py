"""List per-episode tasks with target object + instruction.

Usage:
    python benchmark/scripts/list_mask_tasks.py [results/episode_masks] [--all]

Default lists only the first task of each episode (one-line summary). Pass --all
to list every task.
"""
import json
import os
import sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    root = args[0] if args else "results/episode_masks"

    if not os.path.isdir(root):
        print(f"Not found: {root}")
        sys.exit(1)

    for ep in sorted(os.listdir(root)):
        d = os.path.join(root, ep)
        if not os.path.isdir(d):
            continue
        tasks = sorted(t for t in os.listdir(d) if t.startswith("task_"))
        if not tasks:
            continue
        targets = tasks if show_all else tasks[:1]
        for t in targets:
            meta_path = os.path.join(d, t, "task_meta.json")
            if not os.path.exists(meta_path):
                continue
            m = json.load(open(meta_path))
            print(
                f"{ep[:8]}/{t}  {m['target_object'][:35]:35s}  "
                f"{m['task_instruction'][:60]}"
            )


if __name__ == "__main__":
    main()
