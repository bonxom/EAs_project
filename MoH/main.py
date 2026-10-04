import logging
import random
import sys
from pathlib import Path

import hydra
import numpy as np

ROOT_DIR = str(Path(__file__).resolve().parent)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    # Hydra chdir's to output dir; add project root to sys.path
    sys.path.insert(0, ROOT_DIR)

    workspace_dir = Path.cwd()
    random.seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Project Root: {ROOT_DIR}")

    # Always retain raw responses; separate roles and avoid shared-cache overwrites.
    heu_cache = Path(cfg.heu.get("cache_dir") or "logs/llm") / "heu"
    meta_cache = Path(cfg.meta.get("cache_dir") or "logs/llm") / "meta"
    heu_llm = hydra.utils.instantiate(cfg.heu, cache_dir=str(heu_cache))
    meta_llm = hydra.utils.instantiate(cfg.meta, cache_dir=str(meta_cache))
    logger.info("LLM response caches: heuristic=%s, meta=%s", heu_cache, meta_cache)
    logger.info(f"Using heuristic LLM: {cfg.heu.model}")
    logger.info(f"Using meta LLM: {cfg.meta.model}")

    from moh import MoH
    optimizer = MoH(cfg, ROOT_DIR, heu_llm=heu_llm, meta_llm=meta_llm)
    optimizer.run_meta_optimizer()


if __name__ == "__main__":
    main()
