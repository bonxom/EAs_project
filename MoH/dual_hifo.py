"""Training entrypoint adapter: original MoH plus optional Dual-HiFo guidance."""

from moh import MoH
from utils.dual_hifo import DualHiFoMoHMixin, HiFoConfig
from utils.global_controller import GlobalControllerConfig


class DualHiFoMoH(DualHiFoMoHMixin, MoH):
    def __init__(self, cfg, root_dir, heu_llm, meta_llm=None):
        super().__init__(cfg, root_dir, heu_llm=heu_llm, meta_llm=meta_llm)
        values = dict(cfg.get("dual_hifo", {}))
        values.pop("enabled", None)
        controller_values = dict(values.pop("global_controller", None) or {})
        self.init_dual_hifo(
            HiFoConfig(**values),
            GlobalControllerConfig(**controller_values),
            seed=int(cfg.get("seed", 0)),
        )
