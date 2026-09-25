"""Shared legacy MC bound handling. Y2020 bounds denote baseline multipliers."""
from typing import Any, Optional, Tuple
import math
import re

_Y2020_BOUND_RE = re.compile(r'^\s*y\s*2020[_\- ]?(?P<val>[^\s%]+)\s*%?\s*$', re.IGNORECASE)

def _parse_mc_bound_value(raw: Any) -> Tuple[Optional[float], bool]:
    """Parse MC Min/Max bound value.

    Supports numeric values and Y2020_XX (e.g., Y2020_90 -> 0.90, Y2020_110 -> 1.10).
    Returns (value, is_y2020_ratio).
    """
    if raw is None:
        return None, False
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return None, False
    m = _Y2020_BOUND_RE.match(s)
    if m:
        val_raw = m.group("val").strip()
        try:
            val = float(val_raw)
        except Exception:
            return None, True
        if abs(val) > 2.0:
            val = val / 100.0
        return float(val), True
    try:
        return float(raw), False
    except Exception:
        return None, False


def draw_mc_multiplier(rng, lower, upper, unit):
    """Preserve numeric rate draws; never add one to Y2020 multipliers.

    A mixed absolute/Y2020 interval needs a node-specific baseline. The legacy
    cached solvers cannot resolve it; use the full scenario-effect pipeline.
    """
    lo, lo_relative = _parse_mc_bound_value(lower)
    hi, hi_relative = _parse_mc_bound_value(upper)
    if lo is None or hi is None or not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"Invalid MC bounds: {lower!r}, {upper!r}")
    unit_l = str(unit or '').strip().lower()
    rate = unit_l in {'rate', 'ratio', 'pct', 'percent', 'percentage'}
    if lo_relative != hi_relative:
        if rate or unit_l in {'multiplier', 'factor'}:
            if rate:
                lo = lo if lo_relative else 1.0 + lo
                hi = hi if hi_relative else 1.0 + hi
            lo_relative = hi_relative = True
        else:
            raise ValueError('Mixed absolute/Y2020 MC bounds require the full scenario-effect pipeline')
    lo, hi = sorted((lo, hi))
    draw = float(rng.uniform(lo, hi))
    return max(0.0, draw + (1.0 if rate and not lo_relative else 0.0))
