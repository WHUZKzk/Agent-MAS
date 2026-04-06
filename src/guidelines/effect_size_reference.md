# Effect Size Computation Reference

## Source
Borenstein, M., Hedges, L.V., Higgins, J.P.T., & Rothstein, H.R. (2009). Introduction to Meta-Analysis.
Cochrane Handbook Chapter 10 — Analysing data and undertaking meta-analyses.

## Rules

<!-- TODO: Fill with effect size computation reference for HardNode MathSandbox.

Suggested content:

### Continuous Outcomes
- SMD (Hedges' g): g = (mean_exp - mean_ctrl) / SD_pooled * correction_factor
  SD_pooled = sqrt(((n1-1)*SD1² + (n2-1)*SD2²) / (n1+n2-2))
  Correction J = 1 - 3/(4*(n1+n2-2)-1)   [Hedges small-sample correction]
- MD (Mean Difference): MD = mean_exp - mean_ctrl
  SE_MD = sqrt(SD1²/n1 + SD2²/n2)

### Data Standardization (SE → SD, CI → SD)
- SE to SD: SD = SE * sqrt(n)
- 95% CI to SD: SD = sqrt(n) * (CI_upper - CI_lower) / (2 * 1.96)
- IQR to SD (Wan et al. 2014): SD ≈ IQR / 1.35  [for approximately normal distributions]
- Range to SD (Wan et al. 2014, n-dependent formula): see Table 1

### Dichotomous Outcomes
- OR: OR = (events_exp / (total_exp - events_exp)) / (events_ctrl / (total_ctrl - events_ctrl))
  log(OR) SE = sqrt(1/a + 1/b + 1/c + 1/d)
- RR: RR = (events_exp/total_exp) / (events_ctrl/total_ctrl)

### Validation Rules
- SD must be > 0. If SD <= 0, mark is_valid=False.
- n must be >= 2 for effect size computation.
- CI must satisfy lower < upper.
-->

# TODO
