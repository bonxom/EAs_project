"""Dual-HiFo command-line entry point."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dual_hifo.dual import run_one_size  # noqa: E402
from dual_hifo.llm import build_client  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-HiFo for TSP")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--n-instances", type=int)
    parser.add_argument("--mock", action="store_true", help="use the offline mock LLM")
    parser.add_argument("--model")
    parser.add_argument("--endpoint", "--base-url", dest="base_url")
    parser.add_argument("--api-key")
    parser.add_argument("--output-dir")
    parser.add_argument("--outer-pop", type=int)
    parser.add_argument("--outer-gens", type=int)
    parser.add_argument("--inner-pop", type=int)
    parser.add_argument("--inner-gens", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--neighbor-size", type=int)
    parser.add_argument("--eval-timeout", type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def public_config(cfg: dict) -> dict:
    config = copy.deepcopy(cfg)
    if config.get("llm", {}).get("api_key"):
        config["llm"]["api_key"] = "***"
    return config


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.sizes:
        cfg["sizes"] = args.sizes
    if args.n_instances is not None:
        cfg["n_instances"] = args.n_instances
    if args.random_seed is not None:
        cfg["random_seed"] = args.random_seed
    if args.neighbor_size is not None:
        cfg["neighbor_size"] = args.neighbor_size
    if args.eval_timeout is not None:
        cfg["eval_timeout"] = args.eval_timeout
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.outer_pop is not None:
        cfg["outer"]["pop_size"] = args.outer_pop
    if args.outer_gens is not None:
        cfg["outer"]["n_pop"] = args.outer_gens
    if args.inner_pop is not None:
        cfg["inner"]["pop_size"] = args.inner_pop
    if args.inner_gens is not None:
        cfg["inner"]["n_pop"] = args.inner_gens

    llm = cfg["llm"]
    llm["model"] = args.model or os.environ.get("LLM_MODEL") or llm["model"]
    llm["base_url"] = args.base_url or os.environ.get("LLM_BASE_URL") or llm["base_url"]
    llm["api_key"] = args.api_key or os.environ.get("LLM_API_KEY") or llm.get("api_key", "")
    return cfg


def write_results(run_dir: Path, meta: dict, results: list) -> None:
    summary = {**meta, "results": results}
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Results",
        "",
        "Gap is the mean tour-cost gap versus the optimal tour (%), lower is better.",
        "",
        "| Size | Gap % | Mean cost | Mean optimal | LLM calls |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        if "error" in result:
            lines.append(f"| {result['size']} | FAILED | - | - | - |")
            continue
        best = result["best_heuristic"]
        lines.append(
            f"| {result['size']} | {best['objective']:.4f} | "
            f"{best['mean_cost']:.4f} | {best['mean_optimal_cost']:.4f} | "
            f"{result['llm_calls']} |"
        )
    lines += ["", "## Best heuristics", ""]
    for result in results:
        if "error" in result:
            continue
        lines += [f"### TSP{result['size']}", "",
                  result["best_heuristic"]["idea"], ""]
    (run_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    load_env(ROOT / ".env")
    cfg = apply_overrides(load_config(Path(args.config)), args)

    try:
        client = build_client(cfg["llm"], mock=args.mock)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    output_dir = Path(cfg.get("output_dir", "runs"))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{stamp}_{'mock' if args.mock else 'llm'}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_heuristic = (ROOT / "seeds" / "seed_select_next_node.py").read_text(encoding="utf-8")
    seed_improver = (ROOT / "seeds" / "seed_improver.py").read_text(encoding="utf-8")

    meta = {
        "started_at": stamp,
        "mode": "mock" if args.mock else "llm",
        "model": getattr(client, "model", "unknown"),
        "sizes": cfg["sizes"],
        "config": public_config(cfg),
    }
    print(f"mode={'mock' if args.mock else 'llm'} sizes={cfg['sizes']}")
    print(f"output={run_dir}")

    results, started = [], time.time()
    for size in cfg["sizes"]:
        print(f"TSP{size} ...", flush=True)
        size_started = time.time()
        try:
            result = run_one_size(
                size=size, cfg=cfg, client=client, output_dir=run_dir,
                seed_heuristic=seed_heuristic, seed_improver=seed_improver,
            )
            result["elapsed_sec"] = round(time.time() - size_started, 2)
            print(f"TSP{size}: gap={result['best_heuristic']['objective']:.4f}%")
        except Exception as exc:
            result = {"size": size, "error": f"{type(exc).__name__}: {exc}"}
            print(f"TSP{size}: FAILED ({exc})", file=sys.stderr)
        results.append(result)

    meta["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["elapsed_sec"] = round(time.time() - started, 2)
    write_results(run_dir, meta, results)

    print("\nsize  gap %    cost     optimal  LLM")
    for result in results:
        if "error" in result:
            print(f"{result['size']:>4}  FAILED")
            continue
        best = result["best_heuristic"]
        print(f"{result['size']:>4}  {best['objective']:7.4f}  {best['mean_cost']:7.3f}  "
              f"{best['mean_optimal_cost']:7.3f}  {result['llm_calls']:>3}")
    print(f"\nlogs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
