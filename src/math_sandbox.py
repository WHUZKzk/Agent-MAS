"""
Neuro-Symbolic Math Sandbox — Node 3.4 + Node 3.5.

Spec: docs/07_EXTRACTION_STAGE.md §5 Nodes 3.4 & 3.5

ABSOLUTE RULE: ALL math happens here. LLMs are transcriptionists only.
LLMs MUST NOT compute, derive, or infer any numeric value.

Public API:
  standardize(raw: RawDataPoint) → StandardizedDataPoint
  compute_hedges_g(mean_ig, sd_ig, n_ig, mean_cg, sd_cg, n_cg) → dict
  compute_odds_ratio(events_ig, n_ig, events_cg, n_cg) → dict
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Any

from scipy.stats import norm as _norm

from src.schemas.extraction import RawDataPoint, StandardizedDataPoint

logger = logging.getLogger("autosr.math_sandbox")


# ─────────────────────────────────────────────────────────────────────────────
# Standardization (all 7 data_type routes)
# ─────────────────────────────────────────────────────────────────────────────

def standardize(raw: RawDataPoint) -> StandardizedDataPoint:
    """
    Route a RawDataPoint through the appropriate standardization formula.

    Routes:
      not_reported        → invalid (no data)
      mean_sd             → direct pass-through
      mean_se             → SD = SE × √n
      mean_95ci           → SD = √n × (upper - lower) / 3.92
      median_iqr          → Wan et al. (2014) IQR method
      median_range        → Wan et al. (2014) Range method
      events_total        → direct dichotomous
      percentage_total    → convert pct → events
      or_ci               → store OR directly (no raw counts)

    Assertion failures → StandardizedDataPoint(is_valid=False, …).
    """
    try:
        return _dispatch(raw)
    except (AssertionError, ValueError, ZeroDivisionError) as exc:
        logger.error(
            "[MathSandbox] Standardization failed for data_type=%s: %s",
            raw.data_type, exc,
        )
        return StandardizedDataPoint(
            original_type=raw.data_type,
            is_valid=False,
            validation_notes=str(exc),
        )


def _dispatch(raw: RawDataPoint) -> StandardizedDataPoint:
    dt = raw.data_type

    # ── Route 1: not_reported ────────────────────────────────────────────────
    if dt == "not_reported":
        return StandardizedDataPoint(
            original_type="not_reported",
            is_valid=False,
            validation_notes="Data not reported",
        )

    # ── Route 2: mean_sd ─────────────────────────────────────────────────────
    if dt == "mean_sd":
        mean, sd, n = raw.val1, raw.val2, raw.n
        assert sd is not None and sd > 0, f"SD must be positive, got {sd}"
        assert n is not None and n > 0,   f"N must be positive, got {n}"
        return StandardizedDataPoint(
            original_type=dt,
            mean=mean, sd=sd, n=n,
            standardization_method="direct",
            is_valid=True,
        )

    # ── Route 3: mean_se  →  SD = SE × √n ───────────────────────────────────
    if dt == "mean_se":
        mean = raw.val1
        se   = raw.val2
        n    = raw.n
        assert se is not None and se > 0, f"SE must be positive, got {se}"
        assert n  is not None and n  > 0, f"N must be positive, got {n}"
        sd = se * math.sqrt(n)
        assert sd > 0, f"Computed SD must be positive, got {sd}"
        return StandardizedDataPoint(
            original_type=dt,
            mean=mean, sd=sd, n=n,
            standardization_method="se_to_sd",
            is_valid=True,
        )

    # ── Route 4: mean_95ci  →  SD = √n × (upper - lower) / 3.92 ────────────
    if dt == "mean_95ci":
        mean    = raw.val1
        ci_low  = raw.val2
        ci_high = raw.val3
        n       = raw.n
        assert ci_low  is not None and ci_high is not None, "CI bounds required"
        assert n is not None and n > 0, f"N must be positive, got {n}"
        width = ci_high - ci_low
        assert width > 0, f"CI width must be positive (upper > lower), got {width}"
        sd = math.sqrt(n) * width / 3.92
        assert sd > 0, f"Computed SD must be positive, got {sd}"
        return StandardizedDataPoint(
            original_type=dt,
            mean=mean, sd=sd, n=n,
            standardization_method="ci_to_sd",
            is_valid=True,
        )

    # ── Route 5: median_iqr  →  Wan et al. (2014) IQR Method ────────────────
    if dt == "median_iqr":
        median = raw.val1
        q1     = raw.val2
        q3     = raw.val3
        n      = raw.n
        assert q1 is not None and q3 is not None, "Q1 and Q3 required"
        assert n is not None and n > 0, f"N must be positive, got {n}"
        assert q3 > q1, f"Q3 must be greater than Q1, got Q1={q1} Q3={q3}"

        mean = (q1 + median + q3) / 3.0
        arg  = (0.75 * n - 0.125) / (n + 0.25)
        denom = 2.0 * _norm.ppf(arg)
        assert denom > 0, f"Wan IQR denominator must be positive, got {denom}"
        sd = (q3 - q1) / denom
        assert sd > 0, f"Computed SD must be positive, got {sd}"

        return StandardizedDataPoint(
            original_type=dt,
            mean=mean, sd=sd, n=n,
            standardization_method="wan_iqr",
            is_valid=True,
        )

    # ── Route 6: median_range  →  Wan et al. (2014) Range Method ────────────
    if dt == "median_range":
        median  = raw.val1
        min_val = raw.val2
        max_val = raw.val3
        n       = raw.n
        assert min_val is not None and max_val is not None, "min and max required"
        assert n is not None and n > 0, f"N must be positive, got {n}"
        assert max_val > min_val, (
            f"max must be greater than min, got min={min_val} max={max_val}"
        )

        mean = (min_val + 2.0 * median + max_val) / 4.0
        arg  = (n - 0.375) / (n + 0.25)
        denom = 2.0 * _norm.ppf(arg)
        assert denom > 0, f"Wan Range denominator must be positive, got {denom}"
        sd = (max_val - min_val) / denom
        assert sd > 0, f"Computed SD must be positive, got {sd}"

        return StandardizedDataPoint(
            original_type=dt,
            mean=mean, sd=sd, n=n,
            standardization_method="wan_range",
            is_valid=True,
        )

    # ── Route 7a: events_total  ──────────────────────────────────────────────
    if dt == "events_total":
        events = int(raw.val1)
        total  = raw.n
        return StandardizedDataPoint(
            original_type=dt,
            events=events, total=total,
            standardization_method="direct_dichotomous",
            is_valid=True,
        )

    # ── Route 7b: percentage_total  →  events = round(pct / 100 × n) ────────
    if dt == "percentage_total":
        pct   = raw.val1
        total = raw.n
        assert pct is not None and total is not None and total > 0
        events = round(pct / 100.0 * total)
        return StandardizedDataPoint(
            original_type=dt,
            events=events, total=total,
            standardization_method="percentage_to_events",
            is_valid=True,
        )

    # ── Route 7c: or_ci  →  paper-reported OR passthrough ───────────────────
    if dt == "or_ci":
        return StandardizedDataPoint(
            original_type=dt,
            mean=raw.val1,   # OR stored in mean field per spec
            sd=None,
            n=raw.n,
            standardization_method="direct_or",
            is_valid=True,
            validation_notes="OR+CI reported directly; raw counts unavailable",
        )

    raise ValueError(f"Unknown data_type: {dt}")


# ─────────────────────────────────────────────────────────────────────────────
# Effect Size: Hedges' g  (continuous outcomes)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hedges_g(
    mean_ig: float,
    sd_ig: float,
    n_ig: int,
    mean_cg: float,
    sd_cg: float,
    n_cg: int,
) -> Dict[str, Any]:
    """
    Compute Hedges' g (bias-corrected SMD) for two independent arms.

    Spec: docs/07_EXTRACTION_STAGE.md §5 Node 3.5

    Returns a dict compatible with EffectSizeData fields.
    """
    df = n_ig + n_cg - 2

    pooled_sd = math.sqrt(
        ((n_ig - 1) * sd_ig**2 + (n_cg - 1) * sd_cg**2) / df
    )
    cohens_d = (mean_ig - mean_cg) / pooled_sd

    # Hedges' correction factor
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    hedges_g = cohens_d * J

    # SE of Hedges' g
    se_g = math.sqrt(
        (n_ig + n_cg) / (n_ig * n_cg) +
        hedges_g**2 / (2.0 * (n_ig + n_cg))
    )

    ci_lower = hedges_g - 1.96 * se_g
    ci_upper = hedges_g + 1.96 * se_g

    logger.debug(
        "[MathSandbox] Hedges' g: pooled_sd=%.4f d=%.4f J=%.6f g=%.4f se=%.4f",
        pooled_sd, cohens_d, J, hedges_g, se_g,
    )

    return {
        "effect_measure": "SMD",
        "effect_value": hedges_g,
        "se": se_g,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "computation_method": "hedges_g",
        "is_valid": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Effect Size: Odds Ratio  (dichotomous outcomes)
# ─────────────────────────────────────────────────────────────────────────────

def compute_odds_ratio(
    events_ig: int,
    n_ig: int,
    events_cg: int,
    n_cg: int,
) -> Dict[str, Any]:
    """
    Compute Odds Ratio with 0.5 continuity correction for zero cells.

    Spec: docs/07_EXTRACTION_STAGE.md §5 Node 3.5

    2×2 table:
      a = events_ig,  b = n_ig - events_ig
      c = events_cg,  d = n_cg - events_cg

    If any cell = 0 → add 0.5 to all four cells.

    Returns a dict compatible with EffectSizeData fields.
    """
    a = float(events_ig)
    b = float(n_ig - events_ig)
    c = float(events_cg)
    d = float(n_cg - events_cg)

    # Zero-cell correction
    if a == 0.0 or b == 0.0 or c == 0.0 or d == 0.0:
        logger.info(
            "[MathSandbox] Zero cell detected (a=%g b=%g c=%g d=%g) — "
            "applying 0.5 continuity correction.", a, b, c, d
        )
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    OR = (a * d) / (b * c)
    ln_OR = math.log(OR)
    se_ln_OR = math.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)

    ci_lower = math.exp(ln_OR - 1.96 * se_ln_OR)
    ci_upper = math.exp(ln_OR + 1.96 * se_ln_OR)

    logger.debug(
        "[MathSandbox] OR: a=%g b=%g c=%g d=%g OR=%.4f se_ln=%.4f",
        a, b, c, d, OR, se_ln_OR,
    )

    return {
        "effect_measure": "OR",
        "effect_value": OR,
        "se": se_ln_OR,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "computation_method": "log_or",
        "is_valid": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Direct OR passthrough (when paper reports OR+CI directly)
# ─────────────────────────────────────────────────────────────────────────────

def compute_or_from_reported(
    or_value: float,
    ci_lower: float,
    ci_upper: float,
) -> Dict[str, Any]:
    """
    When the paper directly reports OR + 95%CI (no raw counts available).
    Recover se_ln_OR from the CI width.

    Spec: docs/07_EXTRACTION_STAGE.md §5 Node 3.5 (direct OR case)
    """
    se_ln_OR = (math.log(ci_upper) - math.log(ci_lower)) / 3.92
    return {
        "effect_measure": "OR",
        "effect_value": or_value,
        "se": se_ln_OR,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "computation_method": "direct_or_ci",
        "is_valid": True,
    }
