"""Bundled optimizer sources; never import executable seed modules in the parent."""
from pathlib import Path

_ROOT = Path(__file__).parent
BASIC_SOURCE = (_ROOT / 'basic.py').read_text(encoding='utf-8')
MULTI_TEMPERATURE_SOURCE = (_ROOT / 'multi_temperature.py').read_text(encoding='utf-8')
BEST_PARENT_SOURCE = (_ROOT / 'best_parent.py').read_text(encoding='utf-8')
