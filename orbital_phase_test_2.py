#!/usr/bin/env python3
"""
orbital_phase_test.py
─────────────────────
Test for annual modulation of orbital-phase residuals in J1738+0333.

Hypothesis (PFT sideband signal):
    The amplitude of timing residuals at a given orbital phase should vary
    with a ~1-year period, because the angle between the CMB velocity vector
    and the binary orbital plane changes as Earth orbits the Sun.

    Predicted amplitude: ~420 µs at orbital period, modulated annually.
    Predicted phase of annual maximum: near CMB apex ELON ~ 171.7° (late April).

Method:
    1. Run VLBI-frozen PINT fit (same pipeline as main analysis).
       Binary params (PB, A1, TASC, EPS1, EPS2) remain frozen at par values
       since they are not in the 'free' list — this is deliberate.
    2. For each TOA, compute orbital phase from (MJD - TASC) / PB mod 1.
    3. Bin residuals by orbital phase (N_PHASE_BINS bins).
    4. For each bin, fit an annual sinusoid to the residual vs calendar date.
    5. Report amplitude and phase of annual modulation per orbital phase bin.
    6. Produce diagnostic plots.

Usage:
    python orbital_phase_test.py

Output files (in current directory):
    J1738_orbital_phase_residuals.png   — residuals vs orbital phase, coloured by season
    J1738_orbital_annual_modulation.png — annual sinusoid amplitude per orbital phase bin
    J1738_orbital_phase_test.txt        — numerical results summary
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# ── PINT imports ──────────────────────────────────────────────────────────────
import pint.models as pm
import pint.toa as pt
from pint.models import get_model_and_toas
from pint.fitter import WLSFitter
import astropy.units as u

# ── Import pulsar configuration ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pulsar_data import ( PULSARS, STRIP_RE )

# ── Configuration ─────────────────────────────────────────────────────────────
PULSAR_NAME   = 'J1738+0333'
N_PHASE_BINS  = 12          # orbital phase bins (30° each)
N_SEASON_BINS = 8           # calendar season bins per year (≈6 weeks each)
MIN_TOAS_BIN  = 10          # minimum TOAs per (phase_bin, season) cell
ANNUAL_PERIOD = 365.25      # days

# CMB apex in ecliptic coordinates (for reference phase prediction)
CMB_ELON_DEG  = 171.67      # ecliptic longitude of CMB apex
CMB_ELAT_DEG  = -11.15      # ecliptic latitude of CMB apex

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_PLOT1 = 'J1738_orbital_phase_residuals.png'
OUT_PLOT2 = 'J1738_orbital_annual_modulation.png'
OUT_TEXT  = 'J1738_orbital_phase_test.txt'


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def annual_sinusoid(t_days, A, phi, offset):
    """
    A * sin(2π t / T + phi) + offset  with T = 365.25 days.
    t_days is fractional day-of-year from January 1 (0–365.25),
    consistent with the fractional-year phase convention used throughout.
    phi is in radians.
    """
    return A * np.sin(2 * np.pi * t_days / ANNUAL_PERIOD + phi) + offset


def mjd_to_day_of_year(mjd_array):
    """
    Convert MJD to fractional day-of-year (0–365.25), measured from January 1.

    Uses astropy Time to get the true calendar year fraction, consistent with
    the A*sin(2*pi*t/yr + phi) convention where t=0 is January 1.
    This avoids the MJD-origin offset that would otherwise shift all phases
    by a fixed but large arbitrary amount.
    """
    from astropy.time import Time
    mjd_array = np.asarray(mjd_array)
    t = Time(mjd_array, format='mjd')
    # decimalyear gives e.g. 2017.3456; fractional part * 365.25 = day of year
    dec_year = t.decimalyear
    frac_year = dec_year % 1.0
    return frac_year * ANNUAL_PERIOD


def get_orbital_phase(mjd_array, tasc_mjd, pb_days):
    """
    Compute orbital phase in [0, 1) for each TOA.
    phase = (MJD - TASC) / PB  mod 1
    For ELL1 model, phase=0 is ascending node.
    """
    return ((mjd_array - tasc_mjd) / pb_days) % 1.0


def season_label(day_of_year):
    """Return a rough season label for a fractional day-of-year."""
    seasons = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return seasons[int(day_of_year / ANNUAL_PERIOD * 12) % 12]

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: load config and run VLBI-frozen fit
# ══════════════════════════════════════════════════════════════════════════════

def find_pulsar_config(name):
    for p in PULSARS:
        if p['name'] == name:
            return p
    raise ValueError(f"Pulsar {name} not found in pulsar_data.py")


def build_vlbi_frozen_model(cfg):
    """
    Build a PINT model with VLBI astrometry injected (flag=0)
    and only the parameters in cfg['free'] set to fit=True.
    Mirrors the logic of the main analysis pipeline.

    Strategy: write a modified par file to a NamedTemporaryFile,
    replacing astrometric lines with VLBI values, then load with
    pint.models.get_model().
    """
    import tempfile

    par_path = cfg['par']

    # Read the original par file
    with open(par_path, 'r') as fh:
        lines = fh.readlines()

    # Identify which keywords the VLBI lines will replace
    vlbi_keys = set()
    for vl in cfg['vlbi_lines']:
        parts = vl.strip().split()
        if parts:
            vlbi_keys.add(parts[0].upper())
    vlbi_keys.discard('ECL')   # ECL is a directive, handled separately

    cleaned = [l for l in lines if not STRIP_RE.match(l.strip())]

    # Build filtered line list, dropping lines whose keyword will be replaced
    filtered = []
    for line in cleaned:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            filtered.append(line)
            continue
        key = stripped.split()[0].upper()
        if key not in vlbi_keys:
            filtered.append(line)

    # Append the VLBI replacement lines
    filtered.extend(cfg['vlbi_lines'])

    # Write to a temp file and load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par',
                                     delete=False) as tmp:
        tmp.writelines(filtered)
        tmp_path = tmp.name

    try:
        model = pm.get_model(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Freeze everything, then free only the parameters in cfg['free']
    for pname in model.params:
        try:
            getattr(model, pname).frozen = True
        except Exception:
            pass

    for pname in cfg['free']:
        try:
            getattr(model, pname).frozen = False
        except AttributeError:
            pass  # binary params not present in all pulsars

    return model


def run_fit(cfg):
    """
    Build VLBI-frozen model, load TOAs via get_model_and_toas (matching
    the main pipeline), run WLS fit, return (toas, model, fitter).

    Using get_model_and_toas ensures ephemeris, BIPM version,
    PLANET_SHAPIRO, and all other par-file settings are applied
    consistently without manual flags.
    """
    import tempfile

    print(f"Building VLBI-frozen model ...")
    model = build_vlbi_frozen_model(cfg)

    # Write the frozen model to a temp par so get_model_and_toas can read it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par',
                                     delete=False) as tmp:
        tmp.write(model.as_parfile())
        tmp_par = tmp.name

    try:
        print(f"Loading model and TOAs via get_model_and_toas ...")
        model, toas = get_model_and_toas(
            tmp_par, cfg['tim'],
            allow_T2=True, allow_name_mixing=True
        )
    finally:
        os.unlink(tmp_par)

    print(f"Running WLS fit ({len(toas)} TOAs) ...")
    fitter = WLSFitter(toas, model)
    fitter.fit_toas()

    return toas, model, fitter


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: extract residuals and orbital phases
# ══════════════════════════════════════════════════════════════════════════════

def extract_data(toas, fitter):
    """
    Return arrays:
        mjd         : TOA MJDs (days)
        residuals_us : post-fit residuals in microseconds
        orb_phase   : orbital phase in [0, 1)
        day_of_year : fractional day of year in [0, 365.25)
    """
    residuals_s = fitter.resids.time_resids.to(u.us).value.astype(np.float64)  # µs, cast from float128
    err_us = toas.get_errors().to(u.us).value.astype(np.float64)  # ← add this
    mjd = np.array([t.value for t in toas.get_mjds()])

    # Get binary parameters from the model
    model = fitter.model
    try:
        tasc = model.TASC.value   # MJD
        pb   = model.PB.value     # days
        print(f"  TASC = {tasc:.6f} MJD")
        print(f"  PB   = {pb:.8f} days  ({pb*24:.4f} hours)")
    except AttributeError:
        raise RuntimeError("Model does not contain TASC/PB — is this a binary?")

    orb_phase   = get_orbital_phase(mjd, tasc, pb)
    day_of_year = mjd_to_day_of_year(mjd)

    return mjd, residuals_s, err_us, orb_phase, day_of_year  # ← add err_us


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: bin by orbital phase and fit annual modulation
# ══════════════════════════════════════════════════════════════════════════════

def bin_and_fit(mjd, residuals_us, err_us, orb_phase, day_of_year):  # ← add err_us
    """
    For each orbital phase bin, fit an annual sinusoid to
    the residuals as a function of day_of_year.

    Returns dict with per-bin results.
    """
    phase_edges = np.linspace(0, 1, N_PHASE_BINS + 1)
    phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])

    results = []

    for i in range(N_PHASE_BINS):
        lo, hi = phase_edges[i], phase_edges[i+1]
        mask = (orb_phase >= lo) & (orb_phase < hi)

        t_bin   = day_of_year[mask]
        r_bin   = residuals_us[mask]
        err_bin = err_us[mask]   # ← add alongside t_bin, r_bin
        mjd_bin = mjd[mask]
        n_toas  = mask.sum()

        rec = {
            'phase_center_deg': phase_centers[i] * 360,
            'n_toas': n_toas,
            'annual_amp_us': np.nan,
            'annual_phi_deg': np.nan,
            'annual_offset_us': np.nan,
            'fit_success': False,
            't_bin': t_bin,
            'r_bin': r_bin,
            'mjd_bin': mjd_bin,
        }

        if n_toas < MIN_TOAS_BIN:
            results.append(rec)
            continue

        # Debug: print first bin stats
        if i == 0:
            print(f'    [diag] bin 0: n_toas={n_toas}, t_bin range={t_bin.min():.1f}-{t_bin.max():.1f}, r_bin range={r_bin.min():.1f}-{r_bin.max():.1f} µs')

        # Fit annual sinusoid via linear least squares — no local minima,
        # A is always >= 0, phase is unambiguous.
        # Model: A_cos*cos(wt) + A_sin*sin(wt) + offset
        # => A = sqrt(A_cos^2 + A_sin^2), phi = atan2(A_cos, A_sin)
        try:
            # t_bin is fractional day-of-year from Jan 1 (0-365.25)
            # Model: r = A_sin*sin(wt) + A_cos*cos(wt) + offset
            # => A*sin(wt + phi) where A=sqrt(A_sin^2+A_cos^2),
            #    phi = arctan2(A_cos, A_sin)  [same as vlbi_frozen_analysis.py]
            omega = 2 * np.pi / ANNUAL_PERIOD
            M = np.column_stack([
                np.sin(omega * t_bin),
                np.cos(omega * t_bin),
                np.ones(len(t_bin)),
            ])
            w  = 1.0 / err_bin.astype(np.float64)**2
            Mw = M * np.sqrt(w)[:, None]
            bw = r_bin.astype(np.float64) * np.sqrt(w)
            coeffs, _, _, _ = np.linalg.lstsq(Mw, bw, rcond=None)
            A_sin, A_cos, offset = coeffs

            A   = np.sqrt(A_sin**2 + A_cos**2)
            # phi in [0, 360): arctan2(A_cos, A_sin) gives phase of
            # A*sin(wt + phi) consistent with main analysis convention
            phi = np.degrees(np.arctan2(A_cos, A_sin)) % 360

            # Uncertainty via residual scatter
            fitted = M @ coeffs
            sigma  = np.sqrt(np.sum((r_bin - fitted)**2) / max(len(t_bin) - 3, 1))
            A_err   = sigma * np.sqrt(2 / len(t_bin))
            phi_err = np.degrees(A_err / A) if A > 0 else 180.0

            rec['annual_amp_us']    = A
            rec['annual_amp_err']   = A_err
            rec['annual_phi_deg']   = phi
            rec['annual_phi_err']   = phi_err
            rec['annual_offset_us'] = offset
            rec['fit_success']      = True
        except Exception as e:
            print(f'    WARNING: bin fit failed: {type(e).__name__}: {e}')

        results.append(rec)

    return phase_centers * 360, results  # phase in degrees


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_residuals_vs_phase(mjd, residuals_us, orb_phase, day_of_year):
    """
    Plot residuals vs orbital phase, points coloured by time of year.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # ── Top: all residuals coloured by day-of-year ────────────────────────────
    ax = axes[0]
    sc = ax.scatter(orb_phase * 360, residuals_us,
                    c=day_of_year, cmap='hsv', s=4, alpha=0.5,
                    vmin=0, vmax=365.25)
    cb = fig.colorbar(sc, ax=ax, label='Day of year')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Orbital phase (degrees)')
    ax.set_ylabel('Residual (µs)')
    ax.set_title(f'J1738+0333  —  residuals vs orbital phase, coloured by season')
    ax.set_xlim(0, 360)

    # ── Bottom: same but binned by season ─────────────────────────────────────
    ax = axes[1]
    n_seasons = N_SEASON_BINS
    season_edges = np.linspace(0, ANNUAL_PERIOD, n_seasons + 1)
    colours = cm.plasma(np.linspace(0.1, 0.9, n_seasons))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    phase_bins = np.linspace(0, 1, N_PHASE_BINS + 1)
    phase_c    = 0.5 * (phase_bins[:-1] + phase_bins[1:]) * 360

    for s in range(n_seasons):
        s_lo, s_hi = season_edges[s], season_edges[s+1]
        s_mask = (day_of_year >= s_lo) & (day_of_year < s_hi)
        if s_mask.sum() < 5:
            continue

        # Median residual per phase bin for this season
        med_res = []
        for i in range(N_PHASE_BINS):
            p_lo = phase_bins[i]; p_hi = phase_bins[i+1]
            cell_mask = s_mask & (orb_phase >= p_lo) & (orb_phase < p_hi)
            if cell_mask.sum() >= 3:
                med_res.append(np.median(residuals_us[cell_mask]))
            else:
                med_res.append(np.nan)

        med_res = np.array(med_res)
        mid_day = 0.5 * (s_lo + s_hi)
        month_idx = int(mid_day / ANNUAL_PERIOD * 12) % 12
        label = month_names[month_idx]

        ax.plot(phase_c, med_res, 'o-', color=colours[s],
                ms=5, lw=1.2, label=label, alpha=0.8)

    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Orbital phase (degrees)')
    ax.set_ylabel('Median residual (µs)')
    ax.set_title('Median residual per phase bin, split by season')
    ax.set_xlim(0, 360)
    ax.legend(ncol=4, fontsize=7, loc='upper right')

    fig.tight_layout()
    fig.savefig(OUT_PLOT1, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PLOT1}")


def plot_annual_modulation(phase_centers_deg, results):
    """
    For each orbital phase bin: plot the annual sinusoid amplitude.
    Also plot the best-fit amplitude and phase vs orbital phase.
    """
    good = [r for r in results if r['fit_success']]
    if not good:
        print("No successful sinusoid fits — skipping modulation plot.")
        return

    phase_c = np.array([r['phase_center_deg'] for r in good])
    amps    = np.array([r['annual_amp_us'] for r in good])
    phis    = np.array([r['annual_phi_deg'] for r in good])
    n_toas  = np.array([r['n_toas'] for r in good])

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Panel 1: annual amplitude vs orbital phase
    ax = axes[0]
    ax.bar(phase_c, amps, width=360/N_PHASE_BINS * 0.8,
           color='steelblue', alpha=0.7, edgecolor='navy')
    ax.set_xlabel('Orbital phase (degrees)')
    ax.set_ylabel('Annual modulation amplitude (µs)')
    ax.set_title('Amplitude of annual residual variation per orbital phase bin')
    ax.set_xlim(0, 360)
    ax.axhline(np.mean(amps), color='r', ls='--', lw=1,
               label=f'Mean = {np.mean(amps):.1f} µs')
    ax.legend()

    # Panel 2: annual phase vs orbital phase
    ax = axes[1]
    ax.scatter(phase_c, phis, s=60, color='darkorange', zorder=3)
    # Expected CMB phase (rough guide — the phase of CMB apex in calendar)
    cmb_phase_expected = (CMB_ELON_DEG / 360) * 365.25  # day of year
    cmb_sinusoid_phase = (cmb_phase_expected / ANNUAL_PERIOD) * 360  # degrees
    ax.axhline(cmb_sinusoid_phase % 360, color='g', ls='--', lw=1.5,
               label=f'CMB apex expected phase ≈ {cmb_sinusoid_phase%360:.0f}°')
    ax.set_xlabel('Orbital phase (degrees)')
    ax.set_ylabel('Annual sinusoid phase (degrees)')
    ax.set_title('Phase of annual modulation per orbital phase bin')
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.legend()

    # Panel 3: example — residuals for the highest-amplitude bin with fit overlay
    best_idx = np.argmax(amps)
    best_r   = good[best_idx]
    ax = axes[2]

    t_b = best_r['t_bin']
    r_b = best_r['r_bin']
    t_fine = np.linspace(0, ANNUAL_PERIOD, 300)
    A   = best_r['annual_amp_us']
    phi = np.radians(best_r['annual_phi_deg'])
    C   = best_r['annual_offset_us']
    r_fit = annual_sinusoid(t_fine, A, phi, C)

    ax.scatter(t_b, r_b, s=8, alpha=0.5, color='steelblue',
               label=f'Data (N={best_r["n_toas"]})')
    ax.plot(t_fine, r_fit, 'r-', lw=2,
            label=f'Annual fit: A={A:.1f} µs, φ={best_r["annual_phi_deg"]:.0f}°')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Residual (µs)')
    ax.set_title(f'Orbital phase bin with strongest annual modulation  '
                 f'(phase ≈ {best_r["phase_center_deg"]:.0f}°)')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PLOT2, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PLOT2}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: write text summary
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(phase_centers_deg, results, mjd, residuals_us,
                  orb_phase, day_of_year):
    lines = []
    lines.append("J1738+0333 ORBITAL PHASE ANNUAL MODULATION TEST")
    lines.append("=" * 65)
    lines.append(f"N_PHASE_BINS = {N_PHASE_BINS}  (one bin = {360/N_PHASE_BINS:.0f}°)")
    lines.append(f"MIN_TOAS_BIN = {MIN_TOAS_BIN}")
    lines.append(f"Total TOAs used: {len(mjd)}")
    lines.append(f"MJD range: {mjd.min():.1f} – {mjd.max():.1f}")
    lines.append(f"RMS residual: {np.std(residuals_us):.2f} µs")
    lines.append("")
    lines.append("PFT PREDICTION:")
    lines.append(f"  Sideband amplitude: ~420 µs (at orbital period)")
    lines.append(f"  Annual modulation amplitude: amplitude of orbital residuals")
    lines.append(f"  varies with ~1-year period.")
    lines.append(f"  Expected annual phase: CMB apex ELON={CMB_ELON_DEG}° "
                 f"=> day ≈ {CMB_ELON_DEG/360*365.25:.0f} of year (late April)")
    lines.append("")
    lines.append(f"{'Phase bin':>12} {'N_TOAs':>8} {'Ann. Amp (µs)':>15} "
                 f"{'Ann. Phase (°)':>15} {'Status':>10}")
    lines.append("-" * 65)

    good_amps = []
    for r in results:
        status = "OK" if r['fit_success'] else "too few"
        amp_str = f"{r['annual_amp_us']:.2f} ± {r.get('annual_amp_err', np.nan):.2f}" \
                  if r['fit_success'] else "—"
        phi_str = f"{r['annual_phi_deg']:.1f} ± {r.get('annual_phi_err', np.nan):.1f}" \
                  if r['fit_success'] else "—"
        lines.append(f"{r['phase_center_deg']:>11.1f}° "
                     f"{r['n_toas']:>8d} "
                     f"{amp_str:>15} "
                     f"{phi_str:>15} "
                     f"{status:>10}")
        if r['fit_success']:
            good_amps.append(r['annual_amp_us'])

    lines.append("")
    if good_amps:
        lines.append(f"Mean annual modulation amplitude: {np.mean(good_amps):.2f} µs")
        lines.append(f"Max annual modulation amplitude:  {np.max(good_amps):.2f} µs")
        lines.append(f"Min annual modulation amplitude:  {np.min(good_amps):.2f} µs")
        lines.append("")

        # Are the phases consistent across bins?
        good_phis = [r['annual_phi_deg'] for r in results if r['fit_success']]
        phi_arr   = np.array(good_phis)
        # Circular mean
        phi_rad   = np.radians(phi_arr)
        circ_mean = np.degrees(np.arctan2(np.mean(np.sin(phi_rad)),
                                          np.mean(np.cos(phi_rad)))) % 360
        R_bar     = np.sqrt(np.mean(np.sin(phi_rad))**2 +
                            np.mean(np.cos(phi_rad))**2)
        lines.append(f"Phase coherence across bins:")
        lines.append(f"  Circular mean phase: {circ_mean:.1f}°")
        lines.append(f"  Rayleigh R-bar: {R_bar:.3f}  (1.0 = perfectly coherent)")
        lines.append("")
        lines.append("INTERPRETATION:")
        if R_bar > 0.7:
            lines.append("  Annual phases are COHERENT across orbital phase bins.")
            lines.append("  This is consistent with a sky-wide annual signal (PFT sideband).")
        elif R_bar > 0.4:
            lines.append("  Annual phases show moderate coherence.")
            lines.append("  Marginal evidence for a coherent annual modulation.")
        else:
            lines.append("  Annual phases are incoherent across bins.")
            lines.append("  No evidence for a coherent annual modulation.")

    with open(OUT_TEXT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"Saved: {OUT_TEXT}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("J1738+0333 ORBITAL PHASE ANNUAL MODULATION TEST")
    print("=" * 65)
    print()

    # 1. Get config
    cfg = find_pulsar_config(PULSAR_NAME)
    print(f"Config found: {cfg['name']}")
    print(f"PAR: {cfg['par']}")
    print(f"TIM: {cfg['tim']}")
    print(f"Free params: {cfg['free']}")
    print()

    # 2. Run fit
    toas, model, fitter = run_fit(cfg)
    chi2_r = fitter.resids.chi2_reduced
    print(f"Fit done.  Reduced chi² = {chi2_r:.3f}")
    print()

    # 3. Extract data
    print("Extracting residuals and orbital phases ...")
    mjd, residuals_us, err_us, orb_phase, day_of_year = extract_data(toas, fitter)
    print(f"  {len(mjd)} TOAs extracted")
    print(f"  RMS residual = {np.std(residuals_us):.2f} µs")
    print()

    # 4. Bin and fit
    print(f"Binning into {N_PHASE_BINS} orbital phase bins and fitting annual sinusoids ...")
    phase_centers_deg, results = bin_and_fit(mjd, residuals_us, err_us, orb_phase, day_of_year)

    n_good = sum(r['fit_success'] for r in results)
    print(f"  Successful fits: {n_good} / {N_PHASE_BINS} bins")
    print()

    # 5. Plots
    print("Generating plots ...")
    plot_residuals_vs_phase(mjd, residuals_us, orb_phase, day_of_year)
    plot_annual_modulation(phase_centers_deg, results)

    # 6. Text summary
    write_summary(phase_centers_deg, results, mjd, residuals_us,
                  orb_phase, day_of_year)

    # 7. Console summary
    print()
    print("RESULTS SUMMARY:")
    print("-" * 50)
    good = [r for r in results if r['fit_success']]
    if good:
        amps = [r['annual_amp_us'] for r in good]
        phis = [r['annual_phi_deg'] for r in good]
        phi_rad  = np.radians(phis)
        R_bar    = np.sqrt(np.mean(np.sin(phi_rad))**2 +
                           np.mean(np.cos(phi_rad))**2)
        circ_mean = np.degrees(np.arctan2(
            np.mean(np.sin(phi_rad)),
            np.mean(np.cos(phi_rad)))) % 360

        print(f"Annual modulation amplitude: mean={np.mean(amps):.1f}, "
              f"max={np.max(amps):.1f} µs")
        print(f"Phase coherence (Rayleigh R-bar): {R_bar:.3f}")
        print(f"Circular mean phase: {circ_mean:.1f}°")
        print(f"CMB apex expected annual phase: "
              f"~{CMB_ELON_DEG/360*360:.0f}° "
              f"(day {CMB_ELON_DEG/360*365.25:.0f} of year)")
        print()
        if R_bar > 0.7:
            print("=> COHERENT annual modulation detected across orbital phase bins.")
        else:
            print("=> No coherent annual modulation detected.")
    else:
        print("No successful fits — check TOA count and data quality.")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
