"""Training entrypoint adapter: original MoH plus optional Dual-HiFo guidance."""

from moh import MoH
from utils.dual_hifo import DualHiFoMoHMixin, HiFoConfig


class DualHiFoMoH(DualHiFoMoHMixin, MoH):
    def __init__(self, cfg, root_dir, heu_llm, meta_llm=None):
        super().__init__(cfg, root_dir, heu_llm=heu_llm, meta_llm=meta_llm)
        values = dict(cfg.get("dual_hifo", {}))
        values.pop("enabled", None)
        self.init_dual_hifo(HiFoConfig(**values))
