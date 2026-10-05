"""Dependency-free terminal progress helpers."""

_BLOCKS = "▁▂▃▄▅▆▇█"
_BARS = " ▏▎▍▌▋▊▉█"


def sparkline(values, lower_is_better=True, width=None):
    """Render ``values`` as a one-line unicode sparkline.

    By default smaller values are drawn as taller blocks, so an improving
    (descending) metric still reads left-to-right as "getting better".
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    if width and len(vals) > width:
        vals = vals[-width:]

    lo, hi = min(vals), max(vals)
    if hi == lo:
        return _BLOCKS[len(_BLOCKS) // 2] * len(vals)

    out = []
    for v in vals:
        frac = (v - lo) / (hi - lo)
        if lower_is_better:
            frac = 1.0 - frac
        out.append(_BLOCKS[min(int(frac * len(_BLOCKS)), len(_BLOCKS) - 1)])
    return "".join(out)


def fmt_duration(seconds):
    if seconds is None:
        return "--"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def fmt_gap(value):
    return "  --  " if value is None else f"{value:6.2f}%"
