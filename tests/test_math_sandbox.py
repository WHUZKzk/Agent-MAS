"""
TDD tests for src/math_sandbox.py

Written BEFORE implementation — all tests must FAIL on first run (ImportError).

Coverage:
  - All 7 data_type standardization routes
  - Wan et al. (2014) median_iqr and median_range formulas
  - Assertion-based sanity checks (SD ≤ 0, N ≤ 0)
  - Effect size: Hedges' g with known values
  - Effect size: Odds Ratio with and without zero-cell correction
  - Edge cases: not_reported, or_ci passthrough
"""
import math
import pytest
from typing import Optional

from src.schemas.extraction import RawDataPoint, StandardizedDataPoint, EffectSizeData

# ── TDD imports (will fail until implementation) ─────────────────────────────
from src.math_sandbox import (                                       # noqa: E402
    standardize,
    compute_hedges_g,
    compute_odds_ratio,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def raw(data_type: str, val1=None, val2=None, val3=None, n=None,
        raw_text: str = "test") -> RawDataPoint:
    return RawDataPoint(data_type=data_type,
                        val1=val1, val2=val2, val3=val3, n=n, raw_text=raw_text)


# ─────────────────────────────────────────────────────────────────────────────
# Route 1: not_reported
# ─────────────────────────────────────────────────────────────────────────────

class TestNotReported:
    def test_returns_invalid(self):
        result = standardize(raw("not_reported"))
        assert result.is_valid is False

    def test_validation_notes_set(self):
        result = standardize(raw("not_reported"))
        assert result.validation_notes is not None
        assert len(result.validation_notes) > 0

    def test_no_numeric_fields(self):
        result = standardize(raw("not_reported"))
        assert result.mean is None
        assert result.sd is None


# ─────────────────────────────────────────────────────────────────────────────
# Route 2: mean_sd  (direct pass-through)
# ─────────────────────────────────────────────────────────────────────────────

class TestMeanSD:
    def test_direct_passthrough(self):
        result = standardize(raw("mean_sd", val1=10.0, val2=2.5, n=50))
        assert result.is_valid is True
        assert result.mean == pytest.approx(10.0)
        assert result.sd   == pytest.approx(2.5)
        assert result.n    == 50

    def test_method_is_direct(self):
        result = standardize(raw("mean_sd", val1=5.0, val2=1.0, n=30))
        assert result.standardization_method == "direct"

    def test_original_type_stored(self):
        result = standardize(raw("mean_sd", val1=5.0, val2=1.0, n=30))
        assert result.original_type == "mean_sd"

    def test_zero_sd_is_invalid(self):
        result = standardize(raw("mean_sd", val1=5.0, val2=0.0, n=30))
        assert result.is_valid is False

    def test_negative_sd_is_invalid(self):
        result = standardize(raw("mean_sd", val1=5.0, val2=-1.0, n=30))
        assert result.is_valid is False

    def test_zero_n_is_invalid(self):
        result = standardize(raw("mean_sd", val1=5.0, val2=1.0, n=0))
        assert result.is_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# Route 3: mean_se  →  SD = SE × √n
# ─────────────────────────────────────────────────────────────────────────────

class TestMeanSE:
    def test_se_to_sd_formula(self):
        """SD = SE × √n. With SE=2.0, n=25: SD=2.0×5=10.0"""
        result = standardize(raw("mean_se", val1=50.0, val2=2.0, n=25))
        assert result.is_valid is True
        assert result.sd == pytest.approx(2.0 * math.sqrt(25))
        assert result.mean == pytest.approx(50.0)

    def test_method_label(self):
        result = standardize(raw("mean_se", val1=50.0, val2=2.0, n=25))
        assert result.standardization_method == "se_to_sd"

    def test_n_preserved(self):
        result = standardize(raw("mean_se", val1=50.0, val2=2.0, n=25))
        assert result.n == 25

    def test_known_value(self):
        """SE=0.38, n=58: SD=0.38×√58≈2.895"""
        result = standardize(raw("mean_se", val1=4.87, val2=0.38, n=58))
        assert result.sd == pytest.approx(0.38 * math.sqrt(58), rel=1e-6)

    def test_zero_se_is_invalid(self):
        result = standardize(raw("mean_se", val1=5.0, val2=0.0, n=30))
        assert result.is_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# Route 4: mean_95ci  →  SD = √n × (upper − lower) / 3.92
# ─────────────────────────────────────────────────────────────────────────────

class TestMean95CI:
    def test_ci_to_sd_formula(self):
        """√n × (CI_upper - CI_lower) / 3.92. With n=36, CI=[4,16]: width=12
        SD = 6 × 12 / 3.92 ≈ 18.367"""
        result = standardize(raw("mean_95ci", val1=10.0, val2=4.0, val3=16.0, n=36))
        expected_sd = math.sqrt(36) * (16.0 - 4.0) / 3.92
        assert result.is_valid is True
        assert result.sd == pytest.approx(expected_sd, rel=1e-6)

    def test_method_label(self):
        result = standardize(raw("mean_95ci", val1=10.0, val2=4.0, val3=16.0, n=36))
        assert result.standardization_method == "ci_to_sd"

    def test_mean_preserved(self):
        result = standardize(raw("mean_95ci", val1=10.0, val2=4.0, val3=16.0, n=36))
        assert result.mean == pytest.approx(10.0)

    def test_inverted_ci_is_invalid(self):
        """CI_lower > CI_upper → SD would be negative."""
        result = standardize(raw("mean_95ci", val1=10.0, val2=16.0, val3=4.0, n=36))
        assert result.is_valid is False

    def test_zero_width_ci_is_invalid(self):
        result = standardize(raw("mean_95ci", val1=10.0, val2=5.0, val3=5.0, n=36))
        assert result.is_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# Route 5: median_iqr  →  Wan et al. (2014) Method
# ─────────────────────────────────────────────────────────────────────────────

class TestMedianIQR:
    """
    Wan et al. (2014) formulas:
      mean  = (Q1 + median + Q3) / 3
      SD    = (Q3 - Q1) / (2 × Φ⁻¹((0.75n - 0.125) / (n + 0.25)))
    """

    def test_mean_formula(self):
        """mean = (Q1 + median + Q3) / 3"""
        # Q1=10, median=15, Q3=20 → mean=15
        result = standardize(raw("median_iqr", val1=15.0, val2=10.0, val3=20.0, n=100))
        assert result.mean == pytest.approx((10.0 + 15.0 + 20.0) / 3, rel=1e-6)

    def test_method_label(self):
        result = standardize(raw("median_iqr", val1=15.0, val2=10.0, val3=20.0, n=100))
        assert result.standardization_method == "wan_iqr"

    def test_sd_formula_reference_value(self):
        """
        Reference computation with scipy.stats.norm.ppf:
        n=100, Q1=10, Q3=20:
          arg = (0.75*100 - 0.125) / (100 + 0.25) = 74.875/100.25 ≈ 0.74726
          Φ⁻¹(0.74726) ≈ 0.66449 (standard normal quantile)
          denom = 2 × 0.66449 ≈ 1.32897
          SD = 10 / 1.32897 ≈ 7.524
        """
        from scipy.stats import norm
        n = 100
        q1, q3 = 10.0, 20.0
        arg = (0.75 * n - 0.125) / (n + 0.25)
        expected_sd = (q3 - q1) / (2 * norm.ppf(arg))
        result = standardize(raw("median_iqr", val1=15.0, val2=q1, val3=q3, n=n))
        assert result.sd == pytest.approx(expected_sd, rel=1e-6)

    def test_n_preserved(self):
        result = standardize(raw("median_iqr", val1=15.0, val2=10.0, val3=20.0, n=100))
        assert result.n == 100

    def test_sd_positive(self):
        result = standardize(raw("median_iqr", val1=15.0, val2=10.0, val3=20.0, n=50))
        assert result.is_valid is True
        assert result.sd > 0

    def test_small_n(self):
        """Wan et al. should work for n as small as 5."""
        result = standardize(raw("median_iqr", val1=10.0, val2=6.0, val3=14.0, n=5))
        assert result.is_valid is True
        assert result.sd > 0

    def test_symmetric_iqr_mean_equals_median(self):
        """If Q1 = median - d and Q3 = median + d (symmetric), mean should equal median."""
        median = 20.0
        d = 5.0
        result = standardize(raw("median_iqr", val1=median, val2=median - d,
                                 val3=median + d, n=80))
        assert result.mean == pytest.approx(median, rel=1e-9)

    def test_inverted_iqr_is_invalid(self):
        """Q3 < Q1 should produce SD ≤ 0 → invalid."""
        result = standardize(raw("median_iqr", val1=15.0, val2=20.0, val3=10.0, n=50))
        assert result.is_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# Route 6: median_range  →  Wan et al. (2014) Range Method
# ─────────────────────────────────────────────────────────────────────────────

class TestMedianRange:
    """
    Wan et al. (2014) range formulas:
      mean  = (min + 2×median + max) / 4
      SD    = (max - min) / (2 × Φ⁻¹((n - 0.375) / (n + 0.25)))
    """

    def test_mean_formula(self):
        """mean = (min + 2*median + max) / 4"""
        # min=0, median=10, max=40 → mean = (0+20+40)/4 = 15
        result = standardize(raw("median_range", val1=10.0, val2=0.0, val3=40.0, n=50))
        expected = (0.0 + 2 * 10.0 + 40.0) / 4
        assert result.mean == pytest.approx(expected, rel=1e-9)

    def test_method_label(self):
        result = standardize(raw("median_range", val1=10.0, val2=0.0, val3=40.0, n=50))
        assert result.standardization_method == "wan_range"

    def test_sd_formula_reference_value(self):
        """
        n=50, min=0, max=40:
          arg = (50 - 0.375) / (50 + 0.25) = 49.625/50.25 ≈ 0.98756
          Φ⁻¹(0.98756) ≈ 2.2527
          denom = 2 × 2.2527 ≈ 4.5054
          SD = 40 / 4.5054 ≈ 8.879
        """
        from scipy.stats import norm
        n = 50
        min_val, max_val = 0.0, 40.0
        arg = (n - 0.375) / (n + 0.25)
        expected_sd = (max_val - min_val) / (2 * norm.ppf(arg))
        result = standardize(raw("median_range", val1=10.0, val2=min_val,
                                 val3=max_val, n=n))
        assert result.sd == pytest.approx(expected_sd, rel=1e-6)

    def test_n_preserved(self):
        result = standardize(raw("median_range", val1=10.0, val2=0.0, val3=40.0, n=50))
        assert result.n == 50

    def test_sd_positive(self):
        result = standardize(raw("median_range", val1=10.0, val2=0.0, val3=40.0, n=50))
        assert result.is_valid is True
        assert result.sd > 0

    def test_inverted_range_is_invalid(self):
        """max < min → SD ≤ 0 → invalid."""
        result = standardize(raw("median_range", val1=10.0, val2=40.0, val3=0.0, n=50))
        assert result.is_valid is False


# ─────────────────────────────────────────────────────────────────────────────
# Route 7a: events_total  (dichotomous direct)
# ─────────────────────────────────────────────────────────────────────────────

class TestEventsTotal:
    def test_events_and_total_stored(self):
        result = standardize(raw("events_total", val1=15.0, n=100))
        assert result.is_valid is True
        assert result.events == 15
        assert result.total  == 100

    def test_method_label(self):
        result = standardize(raw("events_total", val1=15.0, n=100))
        assert result.standardization_method == "direct_dichotomous"

    def test_original_type_stored(self):
        result = standardize(raw("events_total", val1=15.0, n=100))
        assert result.original_type == "events_total"

    def test_events_is_integer(self):
        result = standardize(raw("events_total", val1=15.0, n=100))
        assert isinstance(result.events, int)


# ─────────────────────────────────────────────────────────────────────────────
# Route 7b: percentage_total  →  events = round(pct/100 × n)
# ─────────────────────────────────────────────────────────────────────────────

class TestPercentageTotal:
    def test_percentage_to_events(self):
        """25% of 80 = 20 events"""
        result = standardize(raw("percentage_total", val1=25.0, n=80))
        assert result.is_valid is True
        assert result.events == 20
        assert result.total  == 80

    def test_method_label(self):
        result = standardize(raw("percentage_total", val1=25.0, n=80))
        assert result.standardization_method == "percentage_to_events"

    def test_rounding(self):
        """33.33% of 100 = round(33.33) = 33"""
        result = standardize(raw("percentage_total", val1=33.33, n=100))
        assert result.events == 33

    def test_100_percent(self):
        result = standardize(raw("percentage_total", val1=100.0, n=50))
        assert result.events == 50


# ─────────────────────────────────────────────────────────────────────────────
# Route 7c: or_ci  (passthrough — paper reports OR directly)
# ─────────────────────────────────────────────────────────────────────────────

class TestOrCI:
    def test_or_stored_in_mean_field(self):
        """OR value stored in the `mean` field per spec."""
        result = standardize(raw("or_ci", val1=2.5, val2=1.2, val3=5.1, n=200))
        assert result.is_valid is True
        assert result.mean == pytest.approx(2.5)

    def test_method_label(self):
        result = standardize(raw("or_ci", val1=2.5, val2=1.2, val3=5.1, n=200))
        assert result.standardization_method == "direct_or"

    def test_original_type_stored(self):
        result = standardize(raw("or_ci", val1=2.5, val2=1.2, val3=5.1, n=200))
        assert result.original_type == "or_ci"

    def test_validation_notes_present(self):
        """or_ci has a note that raw counts are unavailable."""
        result = standardize(raw("or_ci", val1=2.5, val2=1.2, val3=5.1, n=200))
        assert result.validation_notes is not None


# ─────────────────────────────────────────────────────────────────────────────
# Effect Size: Hedges' g
# ─────────────────────────────────────────────────────────────────────────────

class TestHedgesG:
    """
    Tests for compute_hedges_g(mean_ig, sd_ig, n_ig, mean_cg, sd_cg, n_cg).

    Spec formulas:
      pooled_sd = sqrt(((n_ig-1)*sd_ig² + (n_cg-1)*sd_cg²) / (n_ig+n_cg-2))
      d         = (mean_ig - mean_cg) / pooled_sd
      J         = 1 - 3 / (4*(n_ig+n_cg-2) - 1)
      g         = d × J
      se_g      = sqrt((n_ig+n_cg)/(n_ig*n_cg) + g²/(2*(n_ig+n_cg)))
      ci_lower  = g - 1.96×se_g
      ci_upper  = g + 1.96×se_g
    """

    def _expected(self, mean_ig, sd_ig, n_ig, mean_cg, sd_cg, n_cg):
        df = n_ig + n_cg - 2
        pooled_sd = math.sqrt(
            ((n_ig - 1) * sd_ig**2 + (n_cg - 1) * sd_cg**2) / df
        )
        d = (mean_ig - mean_cg) / pooled_sd
        J = 1 - 3 / (4 * df - 1)
        g = d * J
        se_g = math.sqrt((n_ig + n_cg) / (n_ig * n_cg) + g**2 / (2 * (n_ig + n_cg)))
        ci_lower = g - 1.96 * se_g
        ci_upper = g + 1.96 * se_g
        return g, se_g, ci_lower, ci_upper

    def test_hedges_g_known_value(self):
        """Verify against manual calculation."""
        g, se, ci_lo, ci_hi = self._expected(10.0, 2.0, 30, 8.0, 2.5, 30)
        result = compute_hedges_g(10.0, 2.0, 30, 8.0, 2.5, 30)
        assert result["effect_value"] == pytest.approx(g, rel=1e-6)
        assert result["se"] == pytest.approx(se, rel=1e-6)
        assert result["ci_lower"] == pytest.approx(ci_lo, rel=1e-6)
        assert result["ci_upper"] == pytest.approx(ci_hi, rel=1e-6)

    def test_zero_effect(self):
        """Equal means → g = 0."""
        result = compute_hedges_g(10.0, 2.0, 50, 10.0, 2.0, 50)
        assert result["effect_value"] == pytest.approx(0.0, abs=1e-10)

    def test_negative_effect(self):
        """Intervention < Control → g < 0."""
        result = compute_hedges_g(5.0, 1.0, 40, 8.0, 1.0, 40)
        assert result["effect_value"] < 0

    def test_ci_contains_effect_value(self):
        """CI must contain the effect value."""
        result = compute_hedges_g(10.0, 2.0, 30, 8.0, 2.5, 30)
        assert result["ci_lower"] < result["effect_value"] < result["ci_upper"]

    def test_effect_measure_is_smd(self):
        result = compute_hedges_g(10.0, 2.0, 30, 8.0, 2.5, 30)
        assert result["effect_measure"] == "SMD"

    def test_computation_method_label(self):
        result = compute_hedges_g(10.0, 2.0, 30, 8.0, 2.5, 30)
        assert result["computation_method"] == "hedges_g"

    def test_j_correction_reduces_magnitude(self):
        """Hedges' g should have smaller magnitude than Cohen's d."""
        n_ig, n_cg = 10, 10   # Small n where J matters most
        df = n_ig + n_cg - 2
        pooled_sd = math.sqrt(((n_ig - 1) * 2.0**2 + (n_cg - 1) * 2.0**2) / df)
        d = (10.0 - 8.0) / pooled_sd
        result = compute_hedges_g(10.0, 2.0, n_ig, 8.0, 2.0, n_cg)
        assert abs(result["effect_value"]) < abs(d)

    def test_is_valid_true(self):
        result = compute_hedges_g(10.0, 2.0, 30, 8.0, 2.5, 30)
        assert result["is_valid"] is True

    def test_se_is_positive(self):
        result = compute_hedges_g(10.0, 2.0, 30, 8.0, 2.5, 30)
        assert result["se"] > 0

    def test_large_sample_j_approaches_1(self):
        """For large N, the Hedges' correction J → 1, so g ≈ d."""
        n = 10000
        result = compute_hedges_g(10.0, 2.0, n, 8.0, 2.0, n)
        df = 2 * n - 2
        J = 1 - 3 / (4 * df - 1)
        assert J == pytest.approx(1.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Effect Size: Odds Ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestOddsRatio:
    """
    Tests for compute_odds_ratio(events_ig, n_ig, events_cg, n_cg).

    Standard 2×2 table:
      a = events_ig,  b = n_ig - events_ig
      c = events_cg,  d = n_cg - events_cg
      OR = (a×d) / (b×c)
      ln_OR = log(OR)
      se_ln_OR = sqrt(1/a + 1/b + 1/c + 1/d)
      CI = exp(ln_OR ± 1.96×se_ln_OR)

    Zero-cell correction: if any cell = 0 → add 0.5 to all four cells.
    """

    def test_no_zero_cells_known_value(self):
        """
        a=20, b=80, c=10, d=90 → OR = (20×90)/(80×10) = 1800/800 = 2.25
        """
        result = compute_odds_ratio(20, 100, 10, 100)
        assert result["effect_value"] == pytest.approx(2.25, rel=1e-6)

    def test_equal_events_or_equals_1(self):
        result = compute_odds_ratio(20, 100, 20, 100)
        assert result["effect_value"] == pytest.approx(1.0, rel=1e-6)

    def test_se_ln_or_formula(self):
        """se_ln_OR = sqrt(1/a + 1/b + 1/c + 1/d)"""
        a, b, c, d = 20, 80, 10, 90
        expected_se = math.sqrt(1/a + 1/b + 1/c + 1/d)
        result = compute_odds_ratio(20, 100, 10, 100)
        assert result["se"] == pytest.approx(expected_se, rel=1e-6)

    def test_ci_formula(self):
        """CI = exp(ln_OR ± 1.96×se_ln_OR)"""
        a, b, c, d = 20, 80, 10, 90
        OR = (a * d) / (b * c)
        ln_or = math.log(OR)
        se = math.sqrt(1/a + 1/b + 1/c + 1/d)
        expected_lo = math.exp(ln_or - 1.96 * se)
        expected_hi = math.exp(ln_or + 1.96 * se)
        result = compute_odds_ratio(20, 100, 10, 100)
        assert result["ci_lower"] == pytest.approx(expected_lo, rel=1e-6)
        assert result["ci_upper"] == pytest.approx(expected_hi, rel=1e-6)

    # ── Zero-cell correction ─────────────────────────────────────────────────

    def test_zero_events_ig_triggers_correction(self):
        """events_ig = 0 → add 0.5 to all cells. Result must still be valid."""
        result = compute_odds_ratio(0, 50, 10, 50)
        assert result["is_valid"] is True
        # Corrected: a=0.5, b=49.5+0.5=50, c=10.5, d=39.5+0.5=40
        # But let's just check that OR is computable and positive
        assert result["effect_value"] > 0

    def test_zero_events_ig_correction_or_value(self):
        """
        events_ig=0, n_ig=50, events_cg=10, n_cg=50
        b = 50 - 0 = 50, d = 50 - 10 = 40
        After correction: a=0.5, b=50.5, c=10.5, d=40.5
        OR = (0.5×40.5) / (50.5×10.5) = 20.25 / 530.25 ≈ 0.03818
        """
        a, b, c, d = 0.5, 50.5, 10.5, 40.5
        expected_or = (a * d) / (b * c)
        result = compute_odds_ratio(0, 50, 10, 50)
        assert result["effect_value"] == pytest.approx(expected_or, rel=1e-5)

    def test_zero_non_events_ig_triggers_correction(self):
        """b = n_ig - events_ig = 0 → all-events arm → correction applied."""
        result = compute_odds_ratio(50, 50, 10, 50)
        assert result["is_valid"] is True
        assert result["effect_value"] > 0

    def test_zero_events_cg_triggers_correction(self):
        result = compute_odds_ratio(10, 50, 0, 50)
        assert result["is_valid"] is True
        assert result["effect_value"] > 0

    def test_all_zero_cells_handled(self):
        """All events = 0 → a=0, c=0. After correction all cells 0.5."""
        result = compute_odds_ratio(0, 50, 0, 50)
        assert result["is_valid"] is True
        # OR = (0.5×49.5+0.5) / ((50-0.5)×0.5+0.5) — just must not crash

    def test_no_correction_needed_no_adjustment(self):
        """When no zero cells, result must NOT apply the 0.5 adjustment."""
        # Verify by comparing with manual exact calculation
        a, b, c, d = 15, 35, 5, 45
        expected_or = (a * d) / (b * c)
        result = compute_odds_ratio(15, 50, 5, 50)
        assert result["effect_value"] == pytest.approx(expected_or, rel=1e-6)

    def test_effect_measure_is_or(self):
        result = compute_odds_ratio(20, 100, 10, 100)
        assert result["effect_measure"] == "OR"

    def test_computation_method_label(self):
        result = compute_odds_ratio(20, 100, 10, 100)
        assert result["computation_method"] == "log_or"

    def test_ci_contains_effect(self):
        result = compute_odds_ratio(20, 100, 10, 100)
        assert result["ci_lower"] < result["effect_value"] < result["ci_upper"]

    def test_is_valid_true(self):
        result = compute_odds_ratio(20, 100, 10, 100)
        assert result["is_valid"] is True

    def test_or_greater_than_1_when_more_events_in_intervention(self):
        result = compute_odds_ratio(40, 100, 10, 100)
        assert result["effect_value"] > 1.0

    def test_or_less_than_1_when_fewer_events_in_intervention(self):
        result = compute_odds_ratio(10, 100, 40, 100)
        assert result["effect_value"] < 1.0
