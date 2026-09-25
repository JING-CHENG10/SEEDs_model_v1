"""Validation shared by the in-place linear and logarithmic MC updates."""
import math
from typing import Dict, Mapping, Optional, TypeVar

Key = TypeVar('Key')


def validated_multipliers(values: Optional[Mapping[Key, float]], *, name: str,
                          positive: bool = False) -> Optional[Dict[Key, float]]:
    """Preserve zero for linear/EF factors; reject undefined log factors.

    None leaves a group unchanged; an empty map resets it to baseline.
    Call for every driver before changing solver attributes.
    """
    if values is None:
        return None
    result = {}
    for key, value in values.items():
        try:
            factor = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name} multiplier for {key!r} must be numeric: {value!r}') from exc
        if not math.isfinite(factor) or factor < 0 or (positive and factor == 0):
            domain = 'strictly positive (logarithmic model)' if positive else 'nonnegative'
            raise ValueError(f'{name} multiplier for {key!r} must be finite and {domain}: {value!r}')
        result[key] = factor
    return result
