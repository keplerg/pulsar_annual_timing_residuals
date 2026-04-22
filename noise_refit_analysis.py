"""
noise_refit_analysis.py
=======================
Noise-consistent annual sinusoid detection with VLBI-frozen astrometry.

Motivation
----------
The WLS analysis in vlbi_frozen_analysis.py uses raw TOA uncertainties,
which may not reflect the true noise level of the frozen residuals.  A
reviewer (and Michael Lam) correctly noted that without re-estimating the
noise model after freezing astrometry, the reported FAPs are not
self-consistent.

This script addresses that concern by:
  1. Running the identical VLBI freeze + WLS procedure.
  2. Estimating per-system EFAC (multiplicative noise scale) and a global
     EQUAD (additive white-noise floor) from the frozen residuals via
     maximum-likelihood.
  3. Building effective per-TOA uncertainties:
         σ_eff,i = sqrt( (EFAC_sys × σ_i)² + EQUAD² )
  4. Fitting the annual sinusoid via GLS (noise-weighted least squares)
     using those effective uncertainties.
  5. Computing an F-statistic and its p-value for the annual signal, using
     only the GLS noise covariance.

The J1730−2304 geometric suppression argument (Section 3.4 of the paper)
is independent of the noise model and unaffected by this analysis.  This
script provides noise-consistent amplitude/phase estimates and FAPs to
complement the geometric argument.

Usage
-----
    python noise_refit_analysis.py              # all 7 clean-sample pulsars
    python noise_refit_analysis.py J1730-2304   # single pulsar by name

Output
------
  - Console: per-pulsar WLS vs GLS comparison, noise parameters
  - Console: summary comparison table
  - PNG:     per-pulsar 2-panel plot (WLS residuals + GLS residuals)
  - PNG:     summary bar chart of WLS vs GLS amplitudes
"""

import sys
import os
import re
import tempfile
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import f as f_dist
from astropy.time import Time
import astropy.units as u

from pint.models import get_model_and_toas
from pint.models.parameter import funcParameter
import pint.models as pm
import pint.toa as ptoa
from pint.fitter import WLSFitter
from pint.residuals import Residuals

# Import pulsar configurations from existing module
from pulsar_data import PULSARS, CLEAN_SAMPLE, STRIP_RE

# ── Phase convention ──────────────────────────────────────────────────────────
# Phase is computed as 2π × (decimalyear % 1.0), anchoring zero to January 1
# of each calendar year.  This matches the corrected convention in
# vlbi_frozen_analysis.py (after the MJD-epoch-0 phase bug was fixed).
# Do NOT use 2π/365.25 × MJD — that anchors to MJD epoch 0, giving a ~44°
# systematic offset relative to the Jan-1 convention.

def mjds_to_phase(mjds):
    """Return 2π × (decimalyear % 1.0) for each MJD — Jan-1-anchored phase."""
    dec_year = Time(np.asarray(mjds), format='mjd').decimalyear
    frac     = np.asarray(dec_year) % 1.0
    return 2.0 * np.pi * frac


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — VLBI freeze + WLS  (identical logic to vlbi_frozen_analysis.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_frozen_model_and_toas(pulsar):
    """
    Write a cleaned par file with VLBI astrometry injected and frozen,
    load model + TOAs via PINT, run WLS, return:
        mjds       – MJD array
        res_us     – residuals in µs
        err_us     – nominal TOA uncertainties in µs
        sys_ids    – integer system index per TOA (for per-system EFAC)
        sys_names  – list of system name strings
        fitter     – fitted WLSFitter object (for reference)
    """
    name = pulsar['name']

    # ── Clean par file ────────────────────────────────────────────────────────
    with open(pulsar['par']) as fh:
        lines = fh.readlines()
    cleaned = [l for l in lines if not STRIP_RE.match(l.strip())]
    cleaned = pulsar['vlbi_lines'] + cleaned

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False)
    tmp.writelines(cleaned)
    tmp.flush(); tmp.close()
    tmp_name = tmp.name

    try:
        # ── Load ──────────────────────────────────────────────────────────────
        m, t = get_model_and_toas(
            tmp_name, pulsar['tim'],
            allow_T2=True, allow_name_mixing=True,
        )
        print(f"  {len(t)} TOAs  MJD {t.get_mjds().min():.1f}–{t.get_mjds().max():.1f}")

        # ── Freeze / free parameters ──────────────────────────────────────────
        for par in m.params:
            getattr(m, par).frozen = True

        # Ensure astrometry stays frozen
        for par in ['RAJ','DECJ','ELONG','ELAT','PMRA','PMDEC',
                    'PMELONG','PMELAT','PX','FB0','SINI','M2']:
            if hasattr(m, par):
                getattr(m, par).frozen = True

        # Free only the listed parameters (skip funcParameters)
        for pname in pulsar['free']:
            if hasattr(m, pname):
                p = getattr(m, pname)
                if not isinstance(p, funcParameter):
                    p.frozen = False

        free_params = [p for p in m.params if not getattr(m, p).frozen]
        print(f"  Free ({len(free_params)}): {', '.join(free_params)}")

        # ── WLS fit ───────────────────────────────────────────────────────────
        fitter = WLSFitter(t, m)
        # Guard None-valued parameters
        for par in list(free_params):
            p = getattr(fitter.model, par, None)
            if p is not None and p.value is None:
                p.frozen = True
        fitter.fit_toas()

        # ── Extract outputs ───────────────────────────────────────────────────
        res_obj = fitter.resids
        mjds    = t.get_mjds().value.astype(float)
        res_us  = res_obj.time_resids.to(u.us).value.astype(float)
        err_us  = t.get_errors().to(u.us).value.astype(float)

        # ── System identifiers for per-system EFAC ────────────────────────────
        # Try common NANOGrav flag names in order of preference
        sys_ids, sys_names = _get_system_ids(t)

        rms = np.sqrt(np.mean(res_us**2))
        print(f"  WLS RMS = {rms:.2f} µs   N_sys = {len(sys_names)}")

        return mjds, res_us, err_us, sys_ids, sys_names, fitter

    finally:
        os.unlink(tmp_name)


def _get_system_ids(toas):
    """
    Return (sys_ids, sys_names) where sys_ids is an int array with one
    index per TOA identifying which system it came from.
    Try flag keys 'f', 'sys', 'be', 'fe' in order; fall back to all-zeros.
    """
    for flag_key in ('f', 'sys', 'be', 'fe'):
        try:
            flags, _ = toas.get_flag_value(flag_key)
            if flags and any(f is not None for f in flags):
                names = [str(f) if f is not None else 'unknown' for f in flags]
                unique = sorted(set(names))
                name_to_idx = {n: i for i, n in enumerate(unique)}
                ids = np.array([name_to_idx[n] for n in names], dtype=int)
                return ids, unique
        except Exception:
            continue

    # Fallback: treat all TOAs as one system
    return np.zeros(len(toas), dtype=int), ['all']


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Noise parameter estimation
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_noise_params(res_us, err_us, sys_ids, sys_names, verbose=True):
    """
    Estimate per-system EFAC and global EQUAD via maximum likelihood.

    Model
    -----
    σ_eff,i² = (EFAC_sys(i) × σ_i)² + EQUAD²

    EFAC per system
        Closed-form ML solution: EFAC_s = sqrt( Σ_s r_i²/σ_i²  / N_s )
        (maximises Gaussian log-likelihood for that system with EQUAD=0).
        Floored at 1.0 so we never artificially deflate TOA errors and
        inflate significance.

    EQUAD (global)
        After fixing per-system EFACs, find the EQUAD ≥ 0 that maximises
        the global Gaussian log-likelihood via bounded scalar minimisation.

    Returns
    -------
    efac_per_sys : array  shape (N_sys,)
    equad        : float  (µs)
    sigma_eff    : array  shape (N_toa,)  effective uncertainties
    """
    n_sys   = len(sys_names)
    efacs   = np.ones(n_sys)

    # ── Per-system EFAC ───────────────────────────────────────────────────────
    for s, sname in enumerate(sys_names):
        mask = sys_ids == s
        if mask.sum() < 4:
            if verbose:
                print(f"    {sname}: only {mask.sum()} TOAs, using EFAC=1.0")
            continue
        r_s   = res_us[mask]
        sig_s = err_us[mask]
        chi2_per_n = np.sum((r_s / sig_s)**2) / mask.sum()
        efac_s = max(1.0, np.sqrt(chi2_per_n))
        efacs[s] = efac_s
        if verbose:
            print(f"    {sname:30s}  N={mask.sum():5d}  "
                  f"χ²/N={chi2_per_n:.3f}  EFAC={efac_s:.4f}")

    # Effective sigma after applying per-system EFAC (EQUAD=0 for now)
    sigma_after_efac = np.array([efacs[s] * err_us[i]
                                  for i, s in enumerate(sys_ids)])

    # ── Global EQUAD via bounded 1D ML ────────────────────────────────────────
    equad_max = 3.0 * np.std(res_us)   # generous upper bound

    def neg_log_lik(equad):
        var = sigma_after_efac**2 + equad**2
        return 0.5 * np.sum(res_us**2 / var + np.log(var))

    result = minimize_scalar(
        neg_log_lik,
        bounds=(0.0, equad_max),
        method='bounded',
        options={'xatol': 1e-4},
    )
    equad = result.x if (result.success and result.x > 1e-6) else 0.0

    # Final effective uncertainties
    sigma_eff = np.sqrt(sigma_after_efac**2 + equad**2)

    if verbose:
        print(f"    EQUAD = {equad:.3f} µs")

    return efacs, equad, sigma_eff


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GLS annual sinusoid fit
# ═══════════════════════════════════════════════════════════════════════════════

def gls_annual_fit(mjds, res_us, sigma_eff):
    """
    Fit  r_i = A·sin(ωt_i) + B·cos(ωt_i) + C + ε_i
    where ε_i ~ N(0, σ_eff,i²), using noise-weighted (GLS) least squares.

    Returns
    -------
    amplitude   : float  µs  (half-peak)
    phase_deg   : float  degrees  (convention: A·sin(2πt/yr + φ), t from Jan 1)
    amp_err     : float  µs  (1σ formal uncertainty)
    phase_err   : float  degrees
    f_stat      : float  F-statistic for annual signal vs offset-only null
    p_value     : float  corresponding p-value
    fitted      : array  best-fit sinusoid values at each MJD
    chi2_signal : float  weighted χ² for the signal model
    chi2_null   : float  weighted χ² for the offset-only null model
    """
    N   = len(mjds)
    w   = 1.0 / sigma_eff**2          # weights
    phi = mjds_to_phase(mjds)          # Jan-1-anchored phase (corrected convention)

    # Design matrices
    # Signal model: [sin, cos, 1]
    D_sig  = np.column_stack([np.sin(phi), np.cos(phi), np.ones(N)])
    # Null model: [1]
    # (solved analytically: offset = Σ w_i r_i / Σ w_i)

    # ── GLS signal model ──────────────────────────────────────────────────────
    # θ = (D^T W D)^{-1} D^T W r    W = diag(w)
    DtW    = D_sig.T * w             # shape (3, N)
    DtWD   = DtW @ D_sig             # (3, 3)
    DtWr   = DtW @ res_us            # (3,)

    try:
        cov_theta = np.linalg.inv(DtWD)
    except np.linalg.LinAlgError:
        cov_theta = np.linalg.pinv(DtWD)

    theta  = cov_theta @ DtWr        # [sin_amp, cos_amp, offset]
    sin_c, cos_c, offset = theta

    # Amplitude and phase in A·sin(ωt + φ) convention
    # Phase wrapped to [0°, 360°) to match Table 1 reporting convention
    amplitude = np.sqrt(sin_c**2 + cos_c**2)
    phase_deg = np.degrees(np.arctan2(cos_c, sin_c)) % 360.0

    # Propagate covariance to (amplitude, phase)
    # ∂A/∂s = sin_c/A,  ∂A/∂c = cos_c/A
    # ∂φ/∂s = cos_c/A², ∂φ/∂c = -sin_c/A²
    J_amp   = np.array([sin_c / amplitude, cos_c / amplitude, 0.0])
    J_phase = np.array([cos_c / amplitude**2, -sin_c / amplitude**2, 0.0])
    amp_err   = np.sqrt(J_amp   @ cov_theta @ J_amp)
    phase_err = np.degrees(np.sqrt(J_phase @ cov_theta @ J_phase))

    # Fitted residuals
    fitted = D_sig @ theta

    # ── Weighted χ² for signal and null models ────────────────────────────────
    r_sig      = res_us - fitted
    chi2_signal = np.sum(w * r_sig**2)

    offset_null = np.sum(w * res_us) / np.sum(w)
    r_null      = res_us - offset_null
    chi2_null   = np.sum(w * r_null**2)

    # ── F-statistic ───────────────────────────────────────────────────────────
    # Adding sin + cos components = 2 extra DOF
    dof_signal = N - 3
    if dof_signal < 1:
        f_stat  = np.nan
        p_value = np.nan
    else:
        f_stat  = ((chi2_null - chi2_signal) / 2.0) / (chi2_signal / dof_signal)
        p_value = float(f_dist.sf(f_stat, 2, dof_signal))

    return amplitude, phase_deg, amp_err, phase_err, f_stat, p_value, fitted, chi2_signal, chi2_null


def wls_annual_fit(mjds, res_us):
    """
    Plain (unweighted / equal-weight) annual sinusoid fit for comparison.
    Matches the method used in vlbi_frozen_analysis.py.
    """
    phi    = mjds_to_phase(mjds)       # Jan-1-anchored phase (corrected convention)
    D      = np.column_stack([np.sin(phi), np.cos(phi), np.ones(len(mjds))])
    coeffs, _, _, _ = np.linalg.lstsq(D, res_us, rcond=None)
    sin_c, cos_c, _ = coeffs
    amplitude = np.sqrt(sin_c**2 + cos_c**2)
    phase_deg = np.degrees(np.arctan2(cos_c, sin_c)) % 360.0
    fitted    = D @ coeffs
    return amplitude, phase_deg, fitted


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Per-pulsar analysis driver
# ═══════════════════════════════════════════════════════════════════════════════

def analyse(pulsar):
    name = pulsar['name']
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")

    # ── 1. VLBI freeze + WLS ──────────────────────────────────────────────────
    mjds, res_us, err_us, sys_ids, sys_names, fitter = \
        build_frozen_model_and_toas(pulsar)
    times = Time(mjds, format='mjd').datetime

    # ── 2. WLS annual fit (baseline) ──────────────────────────────────────────
    wls_A, wls_phi, wls_fitted = wls_annual_fit(mjds, res_us)

    rms_before = np.std(res_us)
    rms_after  = np.std(res_us - wls_fitted)
    var_removed = (1 - (rms_after / rms_before)**2) * 100.0

    print(f"\n  WLS annual fit:")
    print(f"    A = {wls_A:.2f} µs   φ = {wls_phi:.1f}°")
    print(f"    RMS {rms_before:.2f} → {rms_after:.2f} µs  ({var_removed:.1f}% variance removed)")

    # ── 3. Noise estimation ───────────────────────────────────────────────────
    print(f"\n  Noise parameter estimation:")
    efacs, equad, sigma_eff = estimate_noise_params(
        res_us, err_us, sys_ids, sys_names, verbose=True
    )

    efac_median = np.median([efacs[sys_ids[i]] for i in range(len(mjds))])
    print(f"    Median EFAC = {efac_median:.4f}   EQUAD = {equad:.3f} µs")
    print(f"    σ_eff range: {sigma_eff.min():.2f}–{sigma_eff.max():.2f} µs  "
          f"(median {np.median(sigma_eff):.2f} µs)")

    # ── 4. GLS annual fit ─────────────────────────────────────────────────────
    gls_A, gls_phi, gls_A_err, gls_phi_err, f_stat, p_val, gls_fitted, \
        chi2_sig, chi2_null = gls_annual_fit(mjds, res_us, sigma_eff)

    rms_gls_before = np.sqrt(np.mean((res_us / sigma_eff)**2))   # normalised
    rms_gls_after  = np.sqrt(np.mean(((res_us - gls_fitted) / sigma_eff)**2))

    print(f"\n  GLS annual fit (noise-consistent):")
    print(f"    A = {gls_A:.2f} ± {gls_A_err:.2f} µs   "
          f"φ = {gls_phi:.1f} ± {gls_phi_err:.1f}°")
    print(f"    F({f_stat:.2f}, 2, {len(mjds)-3})   p = {p_val:.3e}")
    print(f"    χ²_null = {chi2_null:.1f}   χ²_signal = {chi2_sig:.1f}   "
          f"Δχ² = {chi2_null - chi2_sig:.1f}")

    # ── 5. Phase shift ────────────────────────────────────────────────────────
    delta_A   = gls_A   - wls_A
    # Phase difference: wrap to [-180°, +180°] so we report the smallest arc
    delta_phi = ((gls_phi - wls_phi + 180) % 360) - 180

    print(f"\n  WLS → GLS shift:  ΔA = {delta_A:+.2f} µs   Δφ = {delta_phi:+.1f}°")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    _plot_comparison(name, mjds, times, res_us, sigma_eff,
                     wls_A, wls_phi, wls_fitted,
                     gls_A, gls_phi, gls_A_err, gls_phi_err,
                     p_val, efac_median, equad)

    return {
        'name':       name,
        'wls_A':      wls_A,
        'wls_phi':    wls_phi,
        'gls_A':      gls_A,
        'gls_A_err':  gls_A_err,
        'gls_phi':    gls_phi,
        'gls_phi_err':gls_phi_err,
        'f_stat':     f_stat,
        'p_value':    p_val,
        'efac':       efac_median,
        'equad':      equad,
        'delta_A':    delta_A,
        'delta_phi':  delta_phi,
        'n_toa':      len(mjds),
        'n_sys':      len(sys_names),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_comparison(name, mjds, times, res_us, sigma_eff,
                     wls_A, wls_phi, wls_fitted,
                     gls_A, gls_phi, gls_A_err, gls_phi_err,
                     p_val, efac, equad):
    """
    Two-panel comparison plot:
      Top:    WLS residuals with WLS annual fit
      Bottom: same residuals with GLS annual fit + noise-scaled error bars
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{name}  —  WLS vs noise-consistent GLS annual fit", fontsize=11)

    si = np.argsort(mjds)
    t_sorted = np.array(times)[si]

    # ── Top: WLS ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(times, res_us, s=2, alpha=0.4, color='steelblue', zorder=2)
    ax.plot(t_sorted, wls_fitted[si], 'r-', lw=1.5,
            label=f'WLS fit: A={wls_A:.1f} µs, φ={wls_phi:.0f}°')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('Residual (µs)')
    ax.set_title('WLS (unweighted) annual fit')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

    # ── Bottom: GLS ───────────────────────────────────────────────────────────
    # Recompute GLS fitted curve using corrected Jan-1-anchored phase
    phi_sorted = mjds_to_phase(mjds[si])
    gls_curve  = gls_A * np.sin(phi_sorted + np.radians(gls_phi))

    ax = axes[1]
    # Error bars scaled by EFAC+EQUAD (plot a representative subset to avoid clutter)
    n = len(mjds)
    step = max(1, n // 400)
    ax.errorbar(np.array(times)[::step],
                res_us[::step],
                yerr=sigma_eff[::step],
                fmt='none', ecolor='steelblue', alpha=0.3, lw=0.6, zorder=1)
    ax.scatter(times, res_us, s=2, alpha=0.4, color='steelblue', zorder=2)
    ax.plot(t_sorted, gls_curve, 'r-', lw=1.5,
            label=(f'GLS fit: A={gls_A:.1f}±{gls_A_err:.1f} µs, '
                   f'φ={gls_phi:.0f}±{gls_phi_err:.0f}°\n'
                   f'F-test p={p_val:.2e}   '
                   f'EFAC={efac:.3f}  EQUAD={equad:.2f} µs'))
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual (µs)')
    ax.set_title('GLS (noise-weighted) annual fit')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    outfile = f"{name.replace('+', 'p')}_noise_refit.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


def _plot_summary(results):
    """Bar chart comparing WLS and GLS amplitudes across the clean sample."""
    if not results:
        return

    names   = [r['name'] for r in results]
    wls_A   = [r['wls_A'] for r in results]
    gls_A   = [r['gls_A'] for r in results]
    gls_err = [r['gls_A_err'] for r in results]
    x       = np.arange(len(names))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, wls_A, width, label='WLS (unweighted)', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, gls_A, width, label='GLS (noise-weighted)', color='tomato', alpha=0.8,
           yerr=gls_err, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Annual amplitude (µs)')
    ax.set_title('WLS vs noise-consistent GLS annual amplitudes — clean sample')
    ax.legend()
    ax.axhline(0, color='k', lw=0.5)
    plt.tight_layout()
    plt.savefig('noise_refit_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: noise_refit_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results):
    if not results:
        return

    hdr = (f"{'Pulsar':<14} {'WLS A':>8} {'WLS φ':>7} "
           f"{'GLS A':>10} {'GLS φ':>9} "
           f"{'ΔA':>7} {'Δφ':>7} "
           f"{'EFAC':>7} {'EQUAD':>7} "
           f"{'F':>8} {'p-value':>12}")
    sep = '-' * len(hdr)
    print(f"\n{'='*len(hdr)}")
    print("  NOISE REFIT SUMMARY — CLEAN SAMPLE")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print(sep)

    for r in results:
        p_str = f"{r['p_value']:.2e}" if not np.isnan(r['p_value']) else '  nan'
        print(
            f"{r['name']:<14} "
            f"{r['wls_A']:>7.1f}µ "
            f"{r['wls_phi']:>6.0f}° "
            f"{r['gls_A']:>7.1f}±{r['gls_A_err']:.1f}µ "
            f"{r['gls_phi']:>6.0f}±{r['gls_phi_err']:.0f}° "
            f"{r['delta_A']:>+6.1f}µ "
            f"{r['delta_phi']:>+6.0f}° "
            f"{r['efac']:>6.3f} "
            f"{r['equad']:>6.2f}µ "
            f"{r['f_stat']:>7.1f} "
            f"{p_str:>12}"
        )

    print(sep)

    # Phase clustering check on GLS phases
    gls_phases = np.array([r['gls_phi'] for r in results])
    R_bar = np.sqrt(np.mean(np.cos(np.radians(gls_phases)))**2 +
                    np.mean(np.sin(np.radians(gls_phases)))**2)
    N = len(results)
    Z = N * R_bar**2
    # Rayleigh test p-value approximation
    p_rayleigh = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*N))
    print(f"\n  GLS phase clustering (Rayleigh test):  N={N}  "
          f"R̄={R_bar:.3f}  Z={Z:.2f}  p≈{p_rayleigh:.4f}")
    print(f"  GLS phase range: {gls_phases.min():.0f}° – {gls_phases.max():.0f}°  "
          f"(window = {gls_phases.max()-gls_phases.min():.0f}°)\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Filter to clean sample (optionally a single named pulsar from argv)
    requested = set(sys.argv[1:]) if len(sys.argv) > 1 else CLEAN_SAMPLE
    run_list  = [p for p in PULSARS if p['name'] in requested & CLEAN_SAMPLE]

    if not run_list:
        print(f"No matching pulsars found. Available clean sample: {sorted(CLEAN_SAMPLE)}")
        sys.exit(1)

    print(f"\nRunning noise refit analysis on {len(run_list)} pulsar(s):")
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
    if len(results) > 1:
        _plot_summary(results)
