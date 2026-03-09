"""
Compare CORE evaluation results: your base checkpoint(s) vs GPT-2 family.

Reads CSVs from base_eval/ (your models) and eval_bundle/ (pre-computed GPT-2),
then prints a side-by-side table of Centered scores and the CORE metric.
Models are ordered left-to-right by parameter count (smallest first).
Display names include param count in parentheses, e.g. "gpt2 (124M)".

Base eval CSVs should be named base_model_<tag>_<step>.csv (e.g. base_model_d6_001000.csv)
so each depth (d6, d10, ...) gets its own column with correct param counts.
Legacy base_model_<step>.csv is still supported; param count then uses the first
matching checkpoint in sorted tag order.

Usage (from nanochat project root):

    # Use default base dir (~/.cache/nanochat or NANOCHAT_BASE_DIR)
    python -m scripts.compare_base_eval

    # Point at project data dir (where base_eval/ and eval_bundle/ live)
    python -m scripts.compare_base_eval --base-dir data

    # Include Accuracy column in the table
    python -m scripts.compare_base_eval --show-accuracy
"""
import scripts._env_bootstrap  # noqa: F401  # load .env before torch

import csv
import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

from nanochat.common import get_base_dir

# Known param counts for GPT-2 family (millions)
GPT2_PARAM_COUNTS = {
    "openai-community-gpt2": 124,
    "openai-community-gpt2-medium": 355,
    "openai-community-gpt2-large": 774,
    "openai-community-gpt2-xl": 1500,
}


def _parse_base_stem(csv_stem: str) -> Optional[Tuple[Optional[str], int]]:
    """Parse base_model CSV stem. Returns (model_tag, step) or None if not a base_model stem.
    New format: base_model_d6_001000 -> (d6, 1000). Legacy: base_model_001000 -> (None, 1000)."""
    if not csv_stem.startswith("base_model_"):
        return None
    rest = csv_stem[len("base_model_"):]
    # New format: <tag>_<step>, e.g. d6_001000 or d10_000500
    match = re.match(r"^(d\d+)_(\d+)$", rest)
    if match:
        return (match.group(1), int(match.group(2)))
    # Legacy: just step
    step_str = rest.lstrip("0") or "0"
    try:
        return (None, int(step_str))
    except ValueError:
        return None


def _short_name(csv_stem: str) -> str:
    """Human-readable name from CSV filename (no path), without params."""
    parsed = _parse_base_stem(csv_stem)
    if parsed is not None:
        tag, step = parsed
        return f"{tag}@{step}" if tag else f"base@{step}"
    if "gpt2-xl" in csv_stem:
        return "gpt2-xl"
    if "gpt2-large" in csv_stem:
        return "gpt2-large"
    if "gpt2-medium" in csv_stem:
        return "gpt2-medium"
    if "gpt2" in csv_stem:
        return "gpt2"
    return csv_stem


def _format_params(n: int) -> str:
    """Format param count for display (e.g. 124 -> '124M', 1500 -> '1.5B')."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _get_base_model_param_count(base_dir: Path, step: int, model_tag: Optional[str] = None) -> Optional[int]:
    """Get total param count for a base checkpoint from its meta JSON (no model load).
    If model_tag is set, only that tag dir is used; else all tag dirs are tried in sorted order (first match wins)."""
    import torch
    from nanochat.gpt import GPT, GPTConfig

    checkpoints_dir = base_dir / "base_checkpoints"
    if not checkpoints_dir.exists():
        return None
    meta_name = f"meta_{step:06d}.json"
    if model_tag is not None:
        tag_dirs = [checkpoints_dir / model_tag]
    else:
        tag_dirs = sorted([p for p in checkpoints_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    for tag_dir in tag_dirs:
        if not tag_dir.exists() or not tag_dir.is_dir():
            continue
        meta_path = tag_dir / meta_name
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            config_kw = meta.get("model_config")
            if not config_kw:
                return None
            if "window_pattern" not in config_kw:
                config_kw["window_pattern"] = "L"
            config = GPTConfig(**config_kw)
            with torch.device("meta"):
                model = GPT(config)
            return model.num_scaling_params()["total"]
        except Exception:
            continue
    return None


def load_csv(path: Path) -> dict:
    """
    Load a base_eval-style CSV. Returns dict:
      tasks: list of task names (order preserved)
      centered: task -> float
      accuracy: task -> float (if present)
      core: float (CORE metric)
    """
    tasks = []
    centered = {}
    accuracy = {}
    core = None
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            task = row[0].strip().rstrip(",")
            if task == "Task":
                continue
            if task == "CORE":
                try:
                    core = float(row[2].strip()) if len(row) > 2 else None
                except (ValueError, IndexError):
                    pass
                continue
            tasks.append(task)
            try:
                acc_val = row[1].strip().rstrip(",")
                cen_val = row[2].strip() if len(row) > 2 else ""
                accuracy[task] = float(acc_val) if acc_val else None
                centered[task] = float(cen_val) if cen_val else None
            except (ValueError, IndexError):
                accuracy[task] = None
                centered[task] = None
    return {"tasks": tasks, "centered": centered, "accuracy": accuracy, "core": core}


def main():
    parser = argparse.ArgumentParser(description="Compare base_eval results to GPT-2")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Directory containing base_eval/ and eval_bundle/ (default: get_base_dir())",
    )
    parser.add_argument(
        "--show-accuracy",
        action="store_true",
        help="Include Accuracy column per model (default: Centered only)",
    )
    args = parser.parse_args()

    base = Path(args.base_dir) if args.base_dir else Path(get_base_dir())
    base_eval_dir = base / "base_eval"
    eval_bundle_dir = base / "eval_bundle"

    if not base_eval_dir.exists():
        print(f"base_eval dir not found: {base_eval_dir}")
        print("Run base_eval for your model first, or set --base-dir to the dir that contains base_eval/ and eval_bundle/.")
        return
    if not eval_bundle_dir.exists():
        print(f"eval_bundle dir not found: {eval_bundle_dir}")
        return

    # Collect CSVs: base_eval/*.csv and eval_bundle/openai-community-gpt2*.csv
    models = []
    for p in sorted(base_eval_dir.glob("*.csv")):
        models.append(("base_eval", p))
    for p in sorted(eval_bundle_dir.glob("openai-community-gpt2*.csv")):
        models.append(("eval_bundle", p))

    if not models:
        print("No model CSVs found in base_eval/ or eval_bundle/openai-community-gpt2*.csv")
        return

    # Get param count for each model and build display name
    def param_count_for(path: Path, label: str) -> int:
        stem = path.stem
        if label == "eval_bundle" and stem in GPT2_PARAM_COUNTS:
            return GPT2_PARAM_COUNTS[stem] * 1_000_000
        parsed = _parse_base_stem(stem)
        if parsed is not None:
            tag, step = parsed
            n = _get_base_model_param_count(base, step, model_tag=tag)
            return n if n is not None else 0
        return 0

    rows_with_params = []
    for label, path in models:
        data = load_csv(path)
        n_params = param_count_for(path, label)
        short = _short_name(path.stem)
        disp = f"{short} ({_format_params(n_params)})" if n_params else short
        rows_with_params.append((n_params, short, disp, data))

    # Sort by param count (ascending), then by short name for ties
    rows_with_params.sort(key=lambda x: (x[0], x[1]))

    names = [r[2] for r in rows_with_params]
    data_list = [r[3] for r in rows_with_params]

    # Use first model's task order; ensure CORE row last
    all_tasks = data_list[0]["tasks"]
    if "CORE" in all_tasks:
        all_tasks = [t for t in all_tasks if t != "CORE"]
    all_tasks.append("CORE")

    # Table: metric is "Centered" (and optionally "Accuracy")
    show_acc = args.show_accuracy
    col_width = 22
    task_width = 36

    header = "Task".ljust(task_width)
    for n in names:
        header += (" " + n[: col_width - 1].ljust(col_width - 1))
    print(header)
    print("-" * (task_width + len(names) * col_width))

    for task in all_tasks:
        row = task[: task_width - 1].ljust(task_width - 1) + " "
        for data in data_list:
            if task == "CORE":
                val = data["core"]
            else:
                val = data["centered"].get(task) if task in data["centered"] else None
            if val is not None:
                row += f" {val:>10.4f}".ljust(col_width)
            else:
                row += " " + "".ljust(col_width - 1)
        print(row)
        if show_acc and task != "CORE":
            row_acc = "  (acc)".ljust(task_width)
            for data in data_list:
                val = data["accuracy"].get(task) if task in data["accuracy"] else None
                if val is not None:
                    row_acc += f" {val:>10.4f}".ljust(col_width)
                else:
                    row_acc += " " + "".ljust(col_width - 1)
            print(row_acc)

    print()
    print("Legend: Centered = (accuracy - 0.01*random) / (1 - 0.01*random). CORE = mean of centered across tasks.")


if __name__ == "__main__":
    main()
