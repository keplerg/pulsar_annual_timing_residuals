"""
noise_model_comparison.py
=========================
Model-comparison test: does the annual signal in VLBI-frozen residuals
survive a full frequentist red-noise refit?

This script addresses the concern noted in Section 2.3 of the paper
(and in Michael Lam's guidance): whether the annual signal could be
absorbed by a power-law red-noise process whose low-frequency bins
straddle 1 yr⁻¹.

Approach
--------
For each pulsar in the clean sample:

  1.  Obtain VLBI-frozen residuals via the same procedure as
      vlbi_frozen_analysis.py (reusing noise_refit_analysis).

  2.  Estimate white-noise parameters (per-system EFAC, global EQUAD) by
      ML.  These are held fixed during the spectral-model comparison so
      that ΔlogL reflects only the red-noise vs red-noise+annual
      competition.

  3.  Fit two nested Gaussian-process models to the frozen residuals:

          M0:  r = offset          +  red-noise GP
          M1:  r = offset + S(t)   +  red-noise GP
          S(t) = A_sin · sin(2π·frac(t)) + A_cos · cos(2π·frac(t))
          (Jan-1-anchored annual sinusoid, same phase convention as the
          paper's WLS fit.)

      Red noise is a power-law GP with 30 Fourier components at
      frequencies k/T for k = 1…30, covering up to ~2 yr⁻¹ for a
      15-year baseline.  Component k = round(T_yr) sits at exactly
      1 yr⁻¹, so the model has full opportunity to absorb annual power
      into red noise if it can.

  4.  Compute:
          ΔlogL   = logL(M1) − logL(M0)         [> 0 ⇒ M1 preferred]
          LR      = 2 · ΔlogL  ~  χ²(2) under H₀ (Wilks)
          p-value = chi2(2).sf(LR)
          ΔAIC    = AIC(M0) − AIC(M1) = 2·ΔlogL − 4
          ΔBIC    = BIC(M0) − BIC(M1) = 2·ΔlogL − 2·ln(N_toa)
          Best-fit A_annual, φ_annual (+ 1σ) under M1.

Interpretation
--------------
  p < 10⁻³ AND ΔBIC > 10  →  M1 strongly preferred; the annual signal
                             is distinguishable from broadband red
                             noise even when red noise is freely fit.
  p ∈ [10⁻³, 0.01]        →  M1 moderately preferred; some annual
                             power may be shared with red noise.
  p > 0.05                →  M0 preferred; annual component is
                             consistent with absorption into red noise
                             near 1 yr⁻¹.

Caveats
-------
  • White-noise parameters are fixed at their M0-independent ML values.
    A fully joint fit would change absolute logL but not ΔlogL materially.
  • Narrowband TOAs cannot distinguish chromatic (DM) from achromatic
    red noise from residuals alone.  Any residual DM variation beyond
    the fitted DM + DM1 terms will be absorbed by the achromatic
    power-law here.  See Section 4.1 of the paper.
  • A power-law spectrum is a standard but strong assumption.  Pronounced
    non-power-law features (DM events, glitches) could bias this
    comparison and would require further diagnostics.

Usage
-----
    python noise_model_comparison.py                 # all 7 clean-sample pulsars
    python noise_model_comparison.py J1730-2304      # single pulsar by name

Output
------
    Console: per-pulsar M0/M1 best fits, model-comparison stats, and a
             summary table across all pulsars.
    PNG:     per-pulsar diagnostic plot and a summary bar chart.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.linalg import cho_factor, cho_solve
from astropy.time import Time

from pulsar_data import PULSARS
from noise_refit_analysis import (
    build_frozen_model_and_toas,
    estimate_noise_params,
    mjds_to_phase,
    CLEAN_SAMPLE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants and settings
# ─────────────────────────────────────────────────────────────────────────────
YR_TO_S = 365.25 * 86400.0          # seconds per year
FYR_HZ  = 1.0 / YR_TO_S             # 1 yr⁻¹ in Hz
S_TO_US = 1e6

N_FOURIER_COMPONENTS = 30           # PTA standard; covers up to ~2 yr⁻¹

LOG10_A_BOUNDS = (-18.0, -10.0)
# GAMMA_BOUNDS   = (0.5,   6.5)
GAMMA_BOUNDS   = (0.0,    7.0)


# ─────────────────────────────────────────────────────────────────────────────
# Fourier basis for red-noise GP
# ─────────────────────────────────────────────────────────────────────────────
def build_fourier_basis(mjds, n_components):
    """
    Build sin/cos Fourier basis at frequencies k/T for k = 1..n_components,
    where T = span of the TOA set.  Time origin is the first TOA.

    Returns
    -------
    F            : (N_toa, 2·n_components)  basis matrix
                   columns ordered [sin_1, sin_2, ..., sin_K, cos_1, ..., cos_K]
    freqs_per_yr : (n_components,)  frequencies in cycles/yr
    T_yr         : total span (years)
    """
    mjds   = np.asarray(mjds, dtype=float)
    T_yr   = (mjds.max() - mjds.min()) / 365.25
    t_yr   = (mjds - mjds.min()) / 365.25

    freqs_per_yr = np.arange(1, n_components + 1) / T_yr     # cycles/yr
    omega = 2.0 * np.pi * freqs_per_yr[None, :]               # rad/yr
    t_col = t_yr[:, None]

    F = np.hstack([np.sin(omega * t_col), np.cos(omega * t_col)])
    return F, freqs_per_yr, T_yr


# ─────────────────────────────────────────────────────────────────────────────
# Power-law prior variance per Fourier bin
# ─────────────────────────────────────────────────────────────────────────────
def powerlaw_variance_us2(freqs_per_yr, T_yr, log10_A, gamma):
    """
    Variance per Fourier bin (µs²) using the standard enterprise /
    discovery convention

        P(f)·df = (A² / 12π²) · f_yr^(γ-3) · f^(-γ) · df         [s²]

    with A = 10^log10_A (dimensionless), f in Hz, f_yr = 1 cycle/yr in Hz,
    df = 1/T in Hz.  Returned value is multiplied by 10¹² to give µs².

    The same variance is assigned to the sin and cos basis component at
    each frequency, so calling code should duplicate the returned vector
    when building Φ for a basis ordered [sins | cosines].
    """
    f_hz  = freqs_per_yr * FYR_HZ
    df_hz = (1.0 / T_yr) * FYR_HZ
    A2    = 10.0 ** (2.0 * log10_A)

    sigma_sq_s2 = (A2 / (12.0 * np.pi**2)) * \
                  FYR_HZ ** (gamma - 3.0) * f_hz ** (-gamma) * df_hz
    return sigma_sq_s2 * S_TO_US**2


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian-process log-likelihood via Woodbury
# ─────────────────────────────────────────────────────────────────────────────
def gp_log_likelihood(params, res_us, sigma_white_us, F, freqs_per_yr, T_yr,
                      ann_phase, include_annual):
    """
    Gaussian log-likelihood for

        r = mean(params) + z + ε
        z ~ N(0, F Φ Fᵀ)        (red-noise GP, power-law Φ)
        ε ~ N(0, diag(σ_w²))    (white noise, fixed)

    Covariance C = diag(σ_w²) + F Φ Fᵀ handled by the Woodbury identity.

    params layout:
        [log10_A, gamma, offset]                      (include_annual = False)
        [log10_A, gamma, offset, A_sin, A_cos]        (include_annual = True)

    ann_phase : Jan-1-anchored phase 2π·(decyear mod 1)   shape (N_toa,)

    Returns logL (scalar).  Returns -1e12 for parameters outside bounds
    or when the Cholesky factorisation fails.
    """
    log10_A, gamma, offset = params[0], params[1], params[2]

    if not (LOG10_A_BOUNDS[0] <= log10_A <= LOG10_A_BOUNDS[1]):
        return -1e12
    if not (GAMMA_BOUNDS[0]   <= gamma   <= GAMMA_BOUNDS[1]):
        return -1e12

    # Deterministic mean
    if include_annual:
        A_sin, A_cos = params[3], params[4]
        mean = offset + A_sin * np.sin(ann_phase) + A_cos * np.cos(ann_phase)
    else:
        mean = offset

    r = res_us - mean

    # Prior variance per Fourier component — sin and cos share the same value
    phi_half = powerlaw_variance_us2(freqs_per_yr, T_yr, log10_A, gamma)
    if np.any(phi_half <= 0) or np.any(~np.isfinite(phi_half)):
        return -1e12
    phi_full = np.concatenate([phi_half, phi_half])        # matches F column order

    # Woodbury:
    #   C⁻¹   = N⁻¹ − N⁻¹ F (Φ⁻¹ + Fᵀ N⁻¹ F)⁻¹ Fᵀ N⁻¹
    #   log|C| = log|N| + log|Φ| + log|Φ⁻¹ + Fᵀ N⁻¹ F|
    N_inv    = 1.0 / sigma_white_us**2                     # (N_toa,)
    NinvF    = F * N_inv[:, None]                           # (N_toa, 2K)
    FtNinvF  = F.T @ NinvF                                  # (2K, 2K)

    M = FtNinvF.copy()
    idx = np.arange(len(phi_full))
    M[idx, idx] += 1.0 / phi_full

    try:
        L_chol, low = cho_factor(M, lower=True, overwrite_a=False)
    except np.linalg.LinAlgError:
        return -1e12

    FtNinvr = F.T @ (N_inv * r)
    alpha   = cho_solve((L_chol, low), FtNinvr)

    quad_form   = float(np.dot(N_inv * r, r)) - float(np.dot(FtNinvr, alpha))
    log_det_N   = float(np.sum(np.log(sigma_white_us**2)))
    log_det_Phi = float(np.sum(np.log(phi_full)))
    log_det_M   = 2.0 * float(np.sum(np.log(np.diag(L_chol))))
    log_det_C   = log_det_N + log_det_Phi + log_det_M

    N_toa = len(r)
    return -0.5 * (quad_form + log_det_C + N_toa * np.log(2.0 * np.pi))


def neg_log_lik(params, *args):
    """scipy-compatible wrapper for gp_log_likelihood."""
    return -gp_log_likelihood(params, *args)


# ─────────────────────────────────────────────────────────────────────────────
# Fit either M0 or M1
# ─────────────────────────────────────────────────────────────────────────────
def fit_model(res_us, sigma_white_us, F, freqs_per_yr, T_yr, ann_phase,
              include_annual, init_guess):
    """Maximise the GP log-likelihood.  Returns (x_best, logL, cov, result)."""
    bounds = [LOG10_A_BOUNDS, GAMMA_BOUNDS, (None, None)]
    if include_annual:
        bounds += [(None, None), (None, None)]

    result = minimize(
        neg_log_lik, init_guess,
        args=(res_us, sigma_white_us, F, freqs_per_yr, T_yr, ann_phase, include_annual),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-10, 'gtol': 1e-7, 'maxiter': 1000},
    )

    logL = -result.fun

    # Covariance via numerical Hessian
    cov = _numerical_hessian_inverse(
        result.x, res_us, sigma_white_us, F, freqs_per_yr, T_yr, ann_phase, include_annual
    )
    return result.x, logL, cov, result


def _numerical_hessian_inverse(x0, res_us, sigma_white_us, F, freqs_per_yr, T_yr,
                                ann_phase, include_annual):
    """
    Central-difference Hessian of −logL, inverted to give a parameter
    covariance.  Returns NaN matrix if the Hessian is not positive
    definite.  Step sizes are parameter-dependent and chosen so that
    every parameter changes by a comparable amount on the log-likelihood
    surface.
    """
    n = len(x0)
    eps = np.zeros(n)
    eps[0] = 0.02                                    # log10_A
    eps[1] = 0.02                                    # gamma
    eps[2] = max(1e-4, abs(x0[2]) * 1e-3)             # offset (µs)
    if n >= 5:
        eps[3] = max(0.05, abs(x0[3]) * 1e-2)         # A_sin
        eps[4] = max(0.05, abs(x0[4]) * 1e-2)         # A_cos

    H = np.zeros((n, n))
    args = (res_us, sigma_white_us, F, freqs_per_yr, T_yr, ann_phase, include_annual)

    for i in range(n):
        for j in range(i, n):
            xpp = x0.copy(); xpp[i] += eps[i]; xpp[j] += eps[j]
            xpm = x0.copy(); xpm[i] += eps[i]; xpm[j] -= eps[j]
            xmp = x0.copy(); xmp[i] -= eps[i]; xmp[j] += eps[j]
            xmm = x0.copy(); xmm[i] -= eps[i]; xmm[j] -= eps[j]
            fpp = neg_log_lik(xpp, *args)
            fpm = neg_log_lik(xpm, *args)
            fmp = neg_log_lik(xmp, *args)
            fmm = neg_log_lik(xmm, *args)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * eps[i] * eps[j])
            H[j, i] = H[i, j]

    try:
        cov = np.linalg.inv(H)
        if not np.all(np.diag(cov) > 0):
            return np.full_like(H, np.nan)
        return cov
    except np.linalg.LinAlgError:
        return np.full_like(H, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Per-pulsar driver
# ─────────────────────────────────────────────────────────────────────────────
def analyse(pulsar):
    name = pulsar['name']
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")

    # ── 1. VLBI-frozen WLS (reuse) ───────────────────────────────────────────
    print("\n[1] Building VLBI-frozen residuals...")
    mjds, res_us, err_us, sys_ids, sys_names, fitter = \
        build_frozen_model_and_toas(pulsar)

    # ── 2. White-noise ML (reuse Figure 5 pipeline) ──────────────────────────
    print("\n[2] Estimating white-noise parameters (EFAC, EQUAD)...")
    efacs, equad, sigma_white = estimate_noise_params(
        res_us, err_us, sys_ids, sys_names, verbose=False
    )
    efac_median = np.median([efacs[sys_ids[i]] for i in range(len(mjds))])
    print(f"    Median EFAC = {efac_median:.3f}   EQUAD = {equad:.3f} µs")
    print(f"    σ_w range   = {sigma_white.min():.2f}–{sigma_white.max():.2f} µs   "
          f"median = {np.median(sigma_white):.2f} µs")

    # ── 3. Fourier basis ─────────────────────────────────────────────────────
    F, freqs_per_yr, T_yr = build_fourier_basis(mjds, N_FOURIER_COMPONENTS)
    k_annual = int(np.round(T_yr))
    k_annual_bin = k_annual if 1 <= k_annual <= N_FOURIER_COMPONENTS else None
    print(f"\n[3] Fourier basis: {N_FOURIER_COMPONENTS} components, "
          f"f = {freqs_per_yr[0]:.3f}–{freqs_per_yr[-1]:.3f} yr⁻¹, T = {T_yr:.2f} yr")
    if k_annual_bin is not None:
        print(f"    Annual (1 yr⁻¹) maps closest to bin k = {k_annual_bin} "
              f"(f = {freqs_per_yr[k_annual_bin-1]:.4f} yr⁻¹)")

    ann_phase = mjds_to_phase(mjds)     # Jan-1 anchored, matches paper convention

    # ── 4. Fit M0 ────────────────────────────────────────────────────────────
    w = 1.0 / sigma_white**2
    offset_init = float(np.sum(w * res_us) / np.sum(w))
    init_m0 = np.array([-14.0, 4.5, offset_init])

    print("\n[4] Fitting M0 (red noise + offset; no annual)...")
    x0, logL0, cov0, res0 = fit_model(
        res_us, sigma_white, F, freqs_per_yr, T_yr, ann_phase,
        include_annual=False, init_guess=init_m0,
    )
    log10_A_0, gamma_0, offset_0 = x0
    log10_A_0_err = float(np.sqrt(cov0[0, 0])) if np.isfinite(cov0[0, 0]) else np.nan
    gamma_0_err   = float(np.sqrt(cov0[1, 1])) if np.isfinite(cov0[1, 1]) else np.nan
    print(f"    log10_A = {log10_A_0:+.3f} ± {log10_A_0_err:.3f}    "
          f"gamma = {gamma_0:.3f} ± {gamma_0_err:.3f}    "
          f"offset = {offset_0:+.2f} µs")
    print(f"    log L (M0) = {logL0:.3f}    (converged: {res0.success})")

    # ── 5. Fit M1 ────────────────────────────────────────────────────────────
    # Seed (A_sin, A_cos) with a naive GLS using white-noise weights only
    r_minus_off = res_us - offset_0
    D_ann = np.column_stack([np.sin(ann_phase), np.cos(ann_phase)])
    try:
        DtW   = D_ann.T * w
        DtWD  = DtW @ D_ann
        DtWr  = DtW @ r_minus_off
        A_sc  = np.linalg.solve(DtWD, DtWr)
    except np.linalg.LinAlgError:
        A_sc = np.array([0.0, 0.0])

    init_m1 = np.array([log10_A_0, gamma_0, offset_0, A_sc[0], A_sc[1]])

    print("\n[5] Fitting M1 (red noise + deterministic annual)...")
    x1, logL1, cov1, res1 = fit_model(
        res_us, sigma_white, F, freqs_per_yr, T_yr, ann_phase,
        include_annual=True, init_guess=init_m1,
    )
    log10_A_1, gamma_1, offset_1, A_sin_1, A_cos_1 = x1
    log10_A_1_err = float(np.sqrt(cov1[0, 0])) if np.isfinite(cov1[0, 0]) else np.nan
    gamma_1_err   = float(np.sqrt(cov1[1, 1])) if np.isfinite(cov1[1, 1]) else np.nan

    A_ann   = float(np.hypot(A_sin_1, A_cos_1))
    phi_ann = float(np.degrees(np.arctan2(A_cos_1, A_sin_1)) % 360.0)

    # Propagate (A_sin, A_cos) covariance to (A, φ)
    if np.all(np.isfinite(cov1[3:5, 3:5])) and A_ann > 0:
        JA = np.array([A_sin_1 / A_ann,     A_cos_1 / A_ann])
        Jp = np.array([A_cos_1 / A_ann**2, -A_sin_1 / A_ann**2])
        cov_sc = cov1[3:5, 3:5]
        A_err   = float(np.sqrt(JA @ cov_sc @ JA))
        phi_err = float(np.degrees(np.sqrt(Jp @ cov_sc @ Jp)))
    else:
        A_err, phi_err = np.nan, np.nan

    print(f"    log10_A = {log10_A_1:+.3f} ± {log10_A_1_err:.3f}    "
          f"gamma = {gamma_1:.3f} ± {gamma_1_err:.3f}")
    print(f"    A_annual = {A_ann:.2f} ± {A_err:.2f} µs    "
          f"φ = {phi_ann:.1f} ± {phi_err:.1f}°")
    print(f"    log L (M1) = {logL1:.3f}    (converged: {res1.success})")

    # ── 6. Model comparison ──────────────────────────────────────────────────
    N_toa  = len(res_us)
    dlogL  = logL1 - logL0
    LR     = 2.0 * dlogL                           # Wilks ~ χ²(2)
    pval   = float(chi2.sf(LR, df=2)) if LR > 0 else 1.0
    dAIC   = 2.0 * dlogL - 4.0                     # BIC(M0) − BIC(M1): + → M1
    dBIC   = 2.0 * dlogL - 2.0 * np.log(N_toa)

    def _pref(delta):   return "M1" if delta > 0 else "M0"
    def _strength(dBIC):
        if dBIC > 10:  return "very strong"
        if dBIC > 6:   return "strong"
        if dBIC > 2:   return "positive"
        if dBIC > 0:   return "weak"
        if dBIC > -2:  return "weak (favouring M0)"
        if dBIC > -6:  return "positive (favouring M0)"
        return "strong (favouring M0)"

    print("\n[6] Model comparison (M1 − M0):")
    print(f"    ΔlogL   = {dlogL:+.3f}")
    print(f"    LR = 2·ΔlogL = {LR:.3f}    χ²(2) p = {pval:.3e}")
    print(f"    ΔAIC    = {dAIC:+.2f}    ({_pref(dAIC)} preferred)")
    print(f"    ΔBIC    = {dBIC:+.2f}    ({_pref(dBIC)} preferred, {_strength(dBIC)})")

    # ── 7. Diagnostic plot ───────────────────────────────────────────────────
    _plot(name, mjds, res_us, sigma_white, F, freqs_per_yr, T_yr, ann_phase,
          x0, x1, logL0, logL1, pval, dBIC, A_ann, phi_ann, efac_median, equad)

    return {
        'name':         name,
        'N_toa':        N_toa,
        'T_yr':         T_yr,
        'efac_median':  efac_median,
        'equad':        equad,
        'log10_A_M0':   log10_A_0,
        'gamma_M0':     gamma_0,
        'log10_A_M1':   log10_A_1,
        'gamma_M1':     gamma_1,
        'log10_A_M1_err': log10_A_1_err,
        'gamma_M1_err':   gamma_1_err,
        'A_annual':     A_ann,
        'A_err':        A_err,
        'phi_annual':   phi_ann,
        'phi_err':      phi_err,
        'logL_M0':      logL0,
        'logL_M1':      logL1,
        'dlogL':        dlogL,
        'LR':           LR,
        'p_value':      pval,
        'dAIC':         dAIC,
        'dBIC':         dBIC,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def _plot(name, mjds, res_us, sigma_white, F, freqs_per_yr, T_yr, ann_phase,
          x0, x1, logL0, logL1, pval, dBIC, A_ann, phi_ann, efac, equad):
    """
    Three-panel diagnostic plot:
      (a) residuals with M1 best-fit annual overlaid
      (b) residuals after subtracting M1 annual + red-noise posterior mean
      (c) red-noise variance spectrum under M0 and M1
    """
    times = Time(mjds, format='mjd').datetime
    si    = np.argsort(mjds)
    times_sorted = np.array(times)[si]

    # M1 deterministic annual curve
    A_sin, A_cos = x1[3], x1[4]
    ann_curve = A_sin * np.sin(ann_phase) + A_cos * np.cos(ann_phase)

    # Red-noise posterior mean under M1: F (Φ⁻¹ + FᵀN⁻¹F)⁻¹ FᵀN⁻¹ (r − mean)
    N_inv = 1.0 / sigma_white**2
    mean1 = x1[2] + ann_curve
    r1    = res_us - mean1
    phi_half = powerlaw_variance_us2(freqs_per_yr, T_yr, x1[0], x1[1])
    phi_full = np.concatenate([phi_half, phi_half])
    M = F.T @ (F * N_inv[:, None])
    idx = np.arange(len(phi_full))
    M[idx, idx] += 1.0 / phi_full
    try:
        L_chol, low = cho_factor(M, lower=True)
        coef = cho_solve((L_chol, low), F.T @ (N_inv * r1))
        rn_mean = F @ coef
    except np.linalg.LinAlgError:
        rn_mean = np.zeros_like(res_us)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    fig.suptitle(
        f"{name}  —  M0 vs M1 (red noise ± deterministic annual)\n"
        f"ΔBIC = {dBIC:+.1f}   χ² p = {pval:.2e}   "
        f"A_ann = {A_ann:.1f} µs  φ = {phi_ann:.0f}°",
        fontsize=11
    )

    # (a) residuals + M1 annual overlay
    ax = axes[0]
    ax.scatter(times, res_us, s=2, alpha=0.4, color='steelblue', zorder=2)
    ax.plot(times_sorted, (x1[2] + ann_curve)[si], 'r-', lw=1.3,
            label=f'M1 mean: offset + annual   A = {A_ann:.1f} µs, φ = {phi_ann:.0f}°')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel('Residual (µs)')
    ax.set_title('Frozen residuals with M1 deterministic mean')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # (b) residuals after removing M1 mean + red-noise posterior mean
    ax = axes[1]
    r_clean = res_us - mean1 - rn_mean
    ax.scatter(times, r_clean, s=2, alpha=0.4, color='seagreen')
    ax.axhline(0, color='k', lw=0.4)
    rms_clean = float(np.sqrt(np.mean(r_clean**2)))
    ax.set_ylabel('Residual (µs)')
    ax.set_title(f'Residuals minus M1 mean minus M1 red-noise posterior mean    '
                 f'(RMS = {rms_clean:.2f} µs)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # (c) red-noise spectrum
    ax = axes[2]
    phi_M0 = powerlaw_variance_us2(freqs_per_yr, T_yr, x0[0], x0[1])
    phi_M1 = powerlaw_variance_us2(freqs_per_yr, T_yr, x1[0], x1[1])
    ax.loglog(freqs_per_yr, phi_M0, 'o-', color='tomato',  label=f'M0: log10_A = {x0[0]:.2f}, γ = {x0[1]:.2f}')
    ax.loglog(freqs_per_yr, phi_M1, 's-', color='steelblue', label=f'M1: log10_A = {x1[0]:.2f}, γ = {x1[1]:.2f}')
    ax.axvline(1.0, color='gray', ls='--', lw=0.8, label='1 yr⁻¹')
    ax.set_xlabel('Frequency (yr⁻¹)')
    ax.set_ylabel('Red-noise variance per bin (µs²)')
    ax.set_title(f'Red-noise spectrum   (EFAC ≈ {efac:.2f}, EQUAD ≈ {equad:.2f} µs)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    outfile = f"{name.replace('+', 'p')}_noise_model_comparison.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table & bar chart
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_table(results):
    if not results:
        return
    hdr = (f"{'Pulsar':<13} {'N':>5} {'T yr':>6} "
           f"{'log10A M0':>10} {'γ M0':>6} "
           f"{'log10A M1':>10} {'γ M1':>6} "
           f"{'A_ann µs':>10} {'φ°':>6} "
           f"{'ΔlogL':>8} {'χ² p':>11} "
           f"{'ΔAIC':>7} {'ΔBIC':>7}")
    sep = '-' * len(hdr)
    print(f"\n{'=' * len(hdr)}")
    print("  MODEL COMPARISON SUMMARY  (M1 = red noise + deterministic annual)")
    print(f"{'=' * len(hdr)}")
    print(hdr)
    print(sep)
    for r in results:
        print(f"{r['name']:<13} {r['N_toa']:>5d} {r['T_yr']:>6.1f} "
              f"{r['log10_A_M0']:>+10.3f} {r['gamma_M0']:>6.2f} "
              f"{r['log10_A_M1']:>+10.3f} {r['gamma_M1']:>6.2f} "
              f"{r['A_annual']:>6.1f}±{r['A_err']:<3.1f} "
              f"{r['phi_annual']:>6.0f} "
              f"{r['dlogL']:>+8.2f} {r['p_value']:>11.2e} "
              f"{r['dAIC']:>+7.1f} {r['dBIC']:>+7.1f}")
    print(sep)
    print(f"  (ΔAIC, ΔBIC positive ⇒ M1 preferred; ΔBIC > 10 is 'very strong')")

    # Phase clustering on M1 phases
    phis = np.array([r['phi_annual'] for r in results])
    R_bar = float(np.hypot(np.mean(np.cos(np.radians(phis))),
                            np.mean(np.sin(np.radians(phis)))))
    N = len(phis)
    Z = N * R_bar**2
    p_rayleigh = float(np.exp(-Z) * (1.0 + (2.0*Z - Z**2) / (4.0*N)))
    print(f"\n  M1 phase clustering (Rayleigh):  N={N}  R̄={R_bar:.3f}  "
          f"Z={Z:.2f}  p≈{p_rayleigh:.4f}")
    print(f"  Phase range: {phis.min():.0f}°–{phis.max():.0f}°  "
          f"(window {phis.max()-phis.min():.0f}°)\n")


def plot_summary(results):
    if len(results) < 2:
        return
    names = [r['name'] for r in results]
    A     = [r['A_annual'] for r in results]
    A_err = [r['A_err']    for r in results]
    pvals = [r['p_value']  for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(names))

    ax1.bar(x, A, yerr=A_err, color='steelblue', alpha=0.85, capsize=4)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha='right')
    ax1.set_ylabel('A_annual under M1 (µs)')
    ax1.set_title('Annual amplitude recovered with red noise freely fit')

    ax2.bar(x, -np.log10(np.maximum(pvals, 1e-300)), color='tomato', alpha=0.85)
    ax2.axhline(3, color='k', ls='--', lw=0.8, label='p = 10⁻³')
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=30, ha='right')
    ax2.set_ylabel('−log₁₀(p-value)   Wilks χ²(2)')
    ax2.set_title('Model-comparison significance   M1 vs M0')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('noise_model_comparison_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: noise_model_comparison_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    requested = set(sys.argv[1:]) if len(sys.argv) > 1 else CLEAN_SAMPLE
    run_list  = [p for p in PULSARS if p['name'] in requested & CLEAN_SAMPLE]
    if not run_list:
        print(f"No matching pulsars. Clean sample: {sorted(CLEAN_SAMPLE)}")
        sys.exit(1)

    print(f"\nRunning noise-model comparison on {len(run_list)} pulsar(s):")
    for p in run_list:
        print(f"  {p['name']}")

    results = []
    for pulsar in run_list:
        try:
            r = analyse(pulsar)
            results.append(r)
        except Exception as exc:
            print(f"\n  ERROR {pulsar['name']}: {exc}")
            import traceback; traceback.print_exc()

    print_summary_table(results)
    plot_summary(results)
