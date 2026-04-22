"""
PATCHES for simulate_vlbi_annual.py  (revised)
===============================================

Two changes to apply:

  1. Manual red-noise injection — bypasses PINT's broken add_correlated_noise
     path (which crashes for J1713+0747). Red noise is added to residuals
     AFTER the white-noise fit, which slightly overestimates its impact
     (a conservative test of Lam's "stale noise model" concern).

  2. Phase convention fix — report phases in the same convention as the
     paper (t measured from January 1 of each year), so simulation phases
     can be compared directly to Table 1.

Apply by replacing the three indicated functions and updating the call
sites shown at the bottom.

PATCHES v2 for simulate_vlbi_annual.py
=======================================

Fixes three issues found in v1:

  (A) Red-noise parameter reading was catastrophically wrong for some
      pulsars (J0030, J1643 reported ~10^13 µs RMS — nonsense).
      Cause: PINT parameter name/unit doesn't match what the code
      expects. Fix: try more names, add sanity checks, print diagnostics.

  (B) Most pulsars reported "No PLRedNoise in timing model." In NG15,
      noise parameters are commonly stored in a separate .noise JSON
      file alongside the .par file, not in the par itself. Fix: look
      for a matching .noise file and read TNREDAMP/TNREDGAM from it
      if the model doesn't have them.

  (C) J1713+0747 crashed during the pre-fit verification step (NaN in
      SVD), preventing the VLBI-frozen analysis from running at all.
      Fix: make the verification fit non-fatal — log a warning and
      continue with the VLBI-frozen fit (which worked in earlier runs).


PATCHES v3 for simulate_vlbi_annual.py
=======================================

Adds NG15 MCMC chain loading to red_noise_realization().

NANOGrav 15yr delivers noise parameters as MCMC chain files in
    NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/noise/{PULSAR}.nb.chain_1.txt
paired with a parameter-name header in
    NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/noise/{PULSAR}.nb.pars.txt

The chain has one row per MCMC sample, one column per free parameter,
followed by 4 bookkeeping columns (log-posterior, log-likelihood,
acceptance rate, step size).

This patch selects the MAP (maximum-a-posteriori) sample and extracts
the red_noise_log10_A and red_noise_gamma values from it.

Apply by REPLACING the red_noise_realization() function in your script.


# ═════════════════════════════════════════════════════════════════════════════
# No other changes needed to simulate_vlbi_annual.py for this patch.
# The existing call site
#
#     rn_us = red_noise_realization(m_timing, toa_mjds, seed=rn_seed)
#
# will now automatically find and use the NG15 MCMC chain files, picking
# the MAP sample after discarding 25% burn-in.
#
# OPTIONAL: pass pulsar_name=cfg['name'] to be extra explicit:
#
#     rn_us = red_noise_realization(m_timing, toa_mjds, seed=rn_seed,
#                                   pulsar_name=pulsar_cfg['name'])
#
# OPTIONAL: override with M0 values from your own §4.6 analysis:
#
#     M0_PARAMS = {
#         'J0030+0451': {'log10_A': -11.5, 'gamma': 0.2},
#         'J1640+2224': {'log10_A': -11.0, 'gamma': 0.0},
#         'J1643-1224': {'log10_A': -11.7, 'gamma': 0.0},
#         'J1713+0747': {'log10_A': -11.2, 'gamma': 0.0},
#         'J1730-2304': {'log10_A': -11.6, 'gamma': 4.4},
#         'J1738+0333': {'log10_A': -11.3, 'gamma': 0.4},
#         'J2317+1439': {'log10_A': -11.4, 'gamma': 0.0},
#     }
#     rn_us = red_noise_realization(
#         m_timing, toa_mjds, seed=rn_seed,
#         noise_file=M0_PARAMS.get(pulsar_cfg['name']))
#
# Running with both (NG15 chain and M0) and comparing the resulting
# artifact amplitudes gives an independent cross-check — NG15 chain
# = "what NANOGrav actually fit with the free-astrometry posterior"
# and M0 = "what comes out if all low-frequency power is red noise."
# ═════════════════════════════════════════════════════════════════════════════

simulate_vlbi_annual.py
=======================
Simulation control test for the annual residuals paper.

For each pulsar:
  1. Load the real par file (timing astrometry = ground truth)
  2. Generate synthetic TOAs on the same time grid as the real data
     using the TIMING solution → no annual signal by construction
  3. Replace astrometric parameters with VLBI values and freeze them
  4. Refit only F0, F1, DM (+ binary params where needed)
  5. Measure annual amplitude in residuals (LS + sinusoid fit)
  6. Compare to paper-reported amplitudes

If the simulation (which has NO real annual signal) produces annual
residuals of similar amplitude to the paper, ChatGPT's explanation
is confirmed: the signal is an artifact of forcing wrong astrometry.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy, io, tempfile, os, warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='Invalid altitude')
import logging
logging.getLogger('pint.models.troposphere_delay').setLevel(logging.CRITICAL)
logging.getLogger('pint.toa').setLevel(logging.CRITICAL)

from astropy.timeseries import LombScargle
from astropy.time import Time
import astropy.units as u

import pint.models as models
import pint.toa as toa
from pint.fitter import WLSFitter
from pint.simulation import make_fake_toas_uniform
import pint.logging
pint.logging.setup(level="WARNING")

from pulsar_data import ( PULSARS, CLEAN_SAMPLE, NG15_NOISE_DIR, get_residuals, fit_sinusoid, wrap_phase_diff, mjd_to_year, MIN_TOAS )

def equatorial_to_ecliptic_vlbi(vlbi_lines):
    """
    Convert equatorial VLBI lines (RAJ/DECJ/PMRA/PMDEC/PX) to ecliptic
    (ELONG/ELAT/PMELONG/PMELAT/PX) so they can be injected into a par file
    that uses ecliptic coordinates.
    """
    from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
    import astropy.units as u_

    vals = {}
    for line in vlbi_lines:
        parts = line.split()
        if len(parts) >= 2:
            vals[parts[0]] = parts[1]

    ra_str   = vals.get('RAJ',  '00:00:00')
    dec_str  = vals.get('DECJ', '+00:00:00')
    pmra     = float(vals.get('PMRA',  0.0))
    pmdec    = float(vals.get('PMDEC', 0.0))
    px       = vals.get('PX', '0.0')

    coord = SkyCoord(ra=ra_str, dec=dec_str,
                     unit=(u_.hourangle, u_.deg), frame='icrs')
    ecl   = coord.barycentricmeanecliptic

    elon_deg = ecl.lon.deg
    elat_deg = ecl.lat.deg

    # Rotate proper motion: equatorial (pmra*cos(dec), pmdec) -> ecliptic
    eps   = np.radians(23.4392911)
    ra_r  = np.radians(coord.ra.deg)
    dec_r = np.radians(coord.dec.deg)
    elon_r = np.radians(elon_deg)
    elat_r = np.radians(elat_deg)

    p_eq  = np.array([-np.sin(ra_r), np.cos(ra_r), 0.0])
    q_eq  = np.array([-np.sin(dec_r)*np.cos(ra_r),
                      -np.sin(dec_r)*np.sin(ra_r),
                       np.cos(dec_r)])
    p_ecl = np.array([-np.sin(elon_r), np.cos(elon_r), 0.0])
    q_ecl = np.array([-np.sin(elat_r)*np.cos(elon_r),
                      -np.sin(elat_r)*np.sin(elon_r),
                       np.cos(elat_r)])

    # pmra is already pmra*cos(dec) in mas/yr; convert to Cartesian
    pm_vec    = pmra * p_eq + pmdec * q_eq
    pmelong   = np.dot(pm_vec, p_ecl) / np.cos(elat_r)   # mas/yr (not *cos)
    pmelat    = np.dot(pm_vec, q_ecl)

    return [
        f"ELONG   {elon_deg:.10f}  0\n",
        f"ELAT    {elat_deg:.10f}  0\n",
        f"PMELONG {pmelong:.6f}        0\n",
        f"PMELAT  {pmelat:.6f}         0\n",
        f"PX      {px}                 0\n",
        "ECL IERS2010\n",
    ]


def inject_vlbi_and_freeze(m_orig, vlbi_lines):
    """
    Return a new model with VLBI astrometry injected and frozen (FIT=0).
    Handles both equatorial (RAJ/DECJ) and ecliptic (ELONG/ELAT) VLBI inputs.
    Automatically converts equatorial VLBI lines to ecliptic if the par file
    uses ecliptic coordinates, to avoid the AstrometryEquatorial/Ecliptic conflict.
    """
    # Get original par text and detect its coordinate system
    par_text = m_orig.as_parfile()
    par_lines = par_text.split('\n')
    par_uses_ecliptic = any(l.split()[0] in ('ELONG', 'ELAT')
                            for l in par_lines if l.split())

    # Determine coordinate type of the supplied VLBI lines
    vlbi_params = [l.split()[0] for l in vlbi_lines if l.strip()]
    vlbi_is_equatorial = any(p in ('RAJ', 'DECJ', 'PMRA', 'PMDEC')
                             for p in vlbi_params)

    # If par file is ecliptic but VLBI lines are equatorial, convert
    if par_uses_ecliptic and vlbi_is_equatorial:
        vlbi_lines = equatorial_to_ecliptic_vlbi(vlbi_lines)
        vlbi_params = [l.split()[0] for l in vlbi_lines if l.strip()]

    # All astrometry keys to strip from the original par
    astrometry_keys = {'RAJ', 'DECJ', 'PMRA', 'PMDEC',
                       'ELONG', 'ELAT', 'PMELONG', 'PMELAT', 'PX', 'ECL'}

    # Strip all FIT flags from vlbi_lines (force flag=0 = frozen)
    frozen_vlbi = []
    for line in vlbi_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'ECL':
            frozen_vlbi.append(line)
        elif len(parts) >= 2:
            frozen_vlbi.append(f"{parts[0]:<12}{parts[1]:<30} 0\n")

    # Filter out old astrometry lines from par
    new_lines = []
    for line in par_lines:
        parts = line.split()
        if parts and parts[0] in astrometry_keys:
            continue
        new_lines.append(line)

    new_par_text = '\n'.join(new_lines) + '\n' + ''.join(frozen_vlbi)

    # Parse into a PINT model
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        f.write(new_par_text)
        tmp_name = f.name
    try:
        m_new = models.get_model(tmp_name)
    finally:
        os.unlink(tmp_name)

    return m_new


def fit_annual_sinusoid(times_mjd, residuals_us):
    """
    Fit A·sin(2π·t_yr + φ) to residuals, where t_yr is fractional
    calendar year (so phase = 0 corresponds to Jan 1 00:00 UT).

    This matches the phase convention used in the paper's Table 1,
    so simulation phases can be directly compared with observed phases.

    Returns
    -------
    amp_us    : float, amplitude in µs
    phase_deg : float, phase in degrees, wrapped to [0, 360)
    """
    t = Time(np.asarray(times_mjd, dtype=float), format='mjd')
    t_yr = np.asarray(t.decimalyear, dtype=float)

    omega = 2 * np.pi
    A_mat = np.column_stack([
        np.sin(omega * t_yr),
        np.cos(omega * t_yr),
        np.ones_like(t_yr),
    ])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(
            A_mat.astype(np.float64),
            np.asarray(residuals_us, dtype=np.float64),
            rcond=None,
        )
        A_sin, A_cos = float(coeffs[0]), float(coeffs[1])
        amp   = float(np.hypot(A_sin, A_cos))
        phase = float(np.degrees(np.arctan2(A_cos, A_sin)) % 360.0)
        return amp, phase
    except Exception:
        return 0.0, 0.0


def lomb_scargle_annual(times_mjd, residuals_us):
    """LS power at f = 1 yr⁻¹, using fractional calendar year."""
    t = Time(np.asarray(times_mjd, dtype=float), format='mjd')
    t_yr = np.asarray(t.decimalyear, dtype=float)
    ls = LombScargle(t_yr, residuals_us)
    power = ls.power(1.0)
    fap = ls.false_alarm_probability(power)
    return float(power), float(fap)


def strip_noise_model(m):
    """
    Remove only stochastic noise components from the fitting model.
    Deterministic components (SolarWindDispersion, DispersionJump) must
    be kept because they contribute to the synthetic TOAs generated by
    make_fake_toas_uniform — stripping them would create spurious
    residuals that don't reflect astrometric mismatch.
    """
    m2 = copy.deepcopy(m)
    
    stochastic_noise_class_names = {
        'ScaleToaError', 'EcorrNoise', 
        'PLRedNoise', 'PLDMNoise', 'PLChromNoise', 'PLSWNoise',
        'TroposphereDelay',
    }
    
    noise_components = [
        name for name, comp in m2.components.items()
        if (hasattr(comp, 'category') and comp.category == 'noise')
        or type(comp).__name__ in stochastic_noise_class_names
    ]
    
    for name in noise_components:
        try:
            m2.remove_component(name)
        except Exception:
            pass
    return m2


def _load_ng15_noise(parfile_path, pulsar_name=None, verbose=True):
    """
    Load NG15 red-noise parameters from the chain files next to the par file.

    Looks for files matching the NG15 naming convention:
        {noise_dir}/{pulsar}.nb.pars.txt       (parameter name header)
        {noise_dir}/{pulsar}.nb.chain_1.txt    (MCMC samples)

    Returns (log10_A, gamma) from the MAP sample, or (None, None) if not found.
    """
    if parfile_path is None:
        return None, None

    # Derive pulsar name from parfile path if not given
    par_dir = os.path.dirname(parfile_path) or '.'
    par_base = os.path.basename(parfile_path)
    if pulsar_name is None:
        # e.g. "J0030+0451.par" or "J0030+0451_NANOGrav_15yv1.nb.par"
        pulsar_name = par_base.split('.')[0].split('_')[0]

    # NG15 noise dir is usually alongside the par dir or one level up
    candidate_dirs = [
        os.path.join(par_dir, 'noise'),
        os.path.join(par_dir, '..', 'noise'),
        os.path.join(par_dir, '..', '..', 'narrowband', 'noise'),
        par_dir,
    ]
    pars_file  = None
    chain_file = None
    for d in candidate_dirs:
        d = os.path.normpath(d)
        p = os.path.join(d, f"{pulsar_name}.nb.pars.txt")
        c = os.path.join(d, f"{pulsar_name}.nb.chain_1.txt")
        if os.path.exists(p) and os.path.exists(c):
            pars_file  = p
            chain_file = c
            break

    if pars_file is None:
        return None, None

    # Read parameter names
    with open(pars_file) as fh:
        par_names = [ln.strip() for ln in fh if ln.strip()]

    # Find the indices for the two parameters we want
    idx_A = idx_g = None
    for i, name in enumerate(par_names):
        if name.endswith('red_noise_log10_A'):
            idx_A = i
        elif name.endswith('red_noise_gamma'):
            idx_g = i
    if idx_A is None or idx_g is None:
        if verbose:
            print(f"    [RN] pars file {pars_file} has no red_noise_* entries")
        return None, None

    # Read the chain. Last 4 columns are bookkeeping: logpost, loglike, accept, step.
    try:
        chain = np.loadtxt(chain_file)
    except Exception as e:
        if verbose:
            print(f"    [RN] could not load {chain_file}: {e}")
        return None, None

    if chain.ndim == 1:
        chain = chain[np.newaxis, :]

    n_par = len(par_names)
    if chain.shape[1] < n_par + 2:
        if verbose:
            print(f"    [RN] chain has {chain.shape[1]} cols, expected ≥{n_par+2}")
        return None, None

    # MAP sample = row with maximum log-posterior (column n_par)
    # Drop burn-in: first 25% of the chain
    burn = chain.shape[0] // 4
    post_chain = chain[burn:]
    logpost_col = n_par
    imax = int(np.argmax(post_chain[:, logpost_col]))
    log10_A_map = float(post_chain[imax, idx_A])
    gamma_map   = float(post_chain[imax, idx_g])

    if verbose:
        print(f"    [RN] NG15 chain MAP for {pulsar_name}: "
              f"log10_A = {log10_A_map:.3f}, gamma = {gamma_map:.3f}  "
              f"(from {os.path.basename(chain_file)}, "
              f"{post_chain.shape[0]} samples post-burn)")
    return log10_A_map, gamma_map


def red_noise_realization(toa_mjds, pulsar_name, noise_dir,
                          seed=None, n_comp=30, verbose=True,
                          override_params=None):
    """
    Generate a power-law red-noise realization using NG15 MCMC chain files.

    Expects {noise_dir}/{pulsar_name}.nb.pars.txt  (parameter name header)
    and   {noise_dir}/{pulsar_name}.nb.chain_1.txt (MCMC samples).

    Picks the MAP sample (max log-posterior) after discarding 25% burn-in.

    Parameters
    ----------
    toa_mjds       : array of TOA times in MJD
    pulsar_name    : e.g. 'J0030+0451'
    noise_dir      : path to NG15 noise directory
    seed           : RNG seed for reproducibility
    n_comp         : number of Fourier components (default 30, matches §4.6)
    verbose        : print diagnostics
    override_params: optional dict {'log10_A': ..., 'gamma': ...} to bypass
                     chain reading (useful for injecting M0 values)

    Returns
    -------
    rn_us : array of red-noise residuals in µs at the TOA epochs.
            Returns zeros if files not found or sanity checks fail.
    """
    log10_A = None
    gamma   = None
    source  = None

    if isinstance(override_params, dict):
        log10_A = float(override_params.get('log10_A', None)) \
            if override_params.get('log10_A') is not None else None
        gamma = float(override_params.get('gamma', None)) \
            if override_params.get('gamma') is not None else None
        if log10_A is not None and gamma is not None:
            source = "override_params dict"

    if log10_A is None or gamma is None:
        pars_path  = os.path.join(noise_dir, f"{pulsar_name}.nb.pars.txt")
        chain_path = os.path.join(noise_dir, f"{pulsar_name}.nb.chain_1.txt")

        if verbose:
            print(f"    [RN] looking for {pars_path}")
            print(f"    [RN] looking for {chain_path}")

        if not (os.path.exists(pars_path) and os.path.exists(chain_path)):
            if verbose:
                print(f"    [RN] NG15 noise files not found in {noise_dir}")
            return np.zeros_like(toa_mjds, dtype=float)

        # Read parameter name header
        with open(pars_path) as fh:
            par_names = [ln.strip() for ln in fh if ln.strip()]

        idx_A = idx_g = None
        for i, name in enumerate(par_names):
            if name.endswith('red_noise_log10_A'):
                idx_A = i
            elif name.endswith('red_noise_gamma'):
                idx_g = i

        if idx_A is None or idx_g is None:
            if verbose:
                print(f"    [RN] no red_noise_* params in {pars_path}")
            return np.zeros_like(toa_mjds, dtype=float)

        # Load the chain
        try:
            chain = np.loadtxt(chain_path)
        except Exception as e:
            if verbose:
                print(f"    [RN] could not load chain: {e}")
            return np.zeros_like(toa_mjds, dtype=float)

        if chain.ndim == 1:
            chain = chain[np.newaxis, :]

        n_par = len(par_names)
        if chain.shape[1] < n_par + 2:
            if verbose:
                print(f"    [RN] chain has {chain.shape[1]} cols, expected ≥{n_par+2}")
            return np.zeros_like(toa_mjds, dtype=float)

        # MAP sample = row with maximum log-posterior
        # Assume the column right after all the parameters is log-posterior.
        # Drop burn-in (first 25% of samples)
        burn = chain.shape[0] // 4
        post_chain = chain[burn:]
        logpost_col = n_par
        imax = int(np.argmax(post_chain[:, logpost_col]))
        log10_A = float(post_chain[imax, idx_A])
        gamma   = float(post_chain[imax, idx_g])
        source = f"NG15 chain MAP ({post_chain.shape[0]} post-burn samples)"

    # Sanity checks
    if not (-20 <= log10_A <= -8):
        if verbose:
            print(f"    [RN] log10_A = {log10_A} outside [-20, -8]; "
                  f"source was {source}; zeroing")
        return np.zeros_like(toa_mjds, dtype=float)
    if not (0 <= gamma <= 8):
        if verbose:
            print(f"    [RN] gamma = {gamma} outside [0, 8]; "
                  f"source was {source}; zeroing")
        return np.zeros_like(toa_mjds, dtype=float)

    if verbose:
        print(f"    [RN] {pulsar_name}: log10_A = {log10_A:.3f}, "
              f"gamma = {gamma:.3f}  ({source})")

    # Build the realization
    mjds = np.asarray(toa_mjds, dtype=float)
    T_yr = (mjds.max() - mjds.min()) / 365.25
    if T_yr <= 0:
        return np.zeros_like(mjds)

    YR_TO_S = 365.25 * 86400.0
    FYR_HZ  = 1.0 / YR_TO_S
    A_lin   = 10.0 ** log10_A

    freqs_per_yr = np.arange(1, n_comp + 1, dtype=float) / T_yr
    f_hz  = freqs_per_yr * FYR_HZ
    df_hz = (1.0 / T_yr) * FYR_HZ
    phi_s2 = (A_lin**2 / (12 * np.pi**2)) * FYR_HZ**(gamma - 3) * f_hz**(-gamma) * df_hz

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(phi_s2)
    sin_coefs = rng.normal(0.0, sigma)
    cos_coefs = rng.normal(0.0, sigma)

    t_yr = (mjds - mjds.min()) / 365.25
    omega = 2 * np.pi * freqs_per_yr[:, None]
    phase_mat = omega * t_yr[None, :]
    rn_s = (sin_coefs[:, None] * np.sin(phase_mat)
          + cos_coefs[:, None] * np.cos(phase_mat)).sum(axis=0)
    rn_us = rn_s * 1e6

    rms = float(np.std(rn_us))
    if rms > 1000.0 or not np.isfinite(rms):
        if verbose:
            print(f"    [RN] realization RMS = {rms:.1f} µs; zeroing")
        return np.zeros_like(toa_mjds, dtype=float)

    if verbose:
        print(f"    [RN] injected realization RMS = {rms:.2f} µs")

    return rn_us


def run_simulation(pulsar_cfg, add_noise=True, noise_level_us=1.0, rn_seed=None):
    """
    Full simulation pipeline for one pulsar.
    
    Returns dict with timing model, VLBI model, residuals, and annual fit results.
    """
    name = pulsar_cfg['name']
    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")

    # ── Step 1: Load timing model (ground truth) ──────────────────
    print(f"  Loading timing model...")
    m_timing = models.get_model(pulsar_cfg['par'])

    # ── Step 2: Load real TOAs to get the time grid ───────────────
    print(f"  Loading real TOAs for time grid...")
    real_toas = toa.get_TOAs(pulsar_cfg['tim'])
    mjds = real_toas.get_mjds().value
    t_start = mjds.min()
    t_end   = mjds.max()
    span_yr = (t_end - t_start) / 365.25
    n_toas  = len(mjds)
    print(f"  Span: {span_yr:.1f} yr,  N_TOAs: {n_toas}")

    # ── Step 3: Generate synthetic TOAs using TIMING model ────────
    # Use ~biweekly cadence matching the real dataset
    n_fake = max(200, int(span_yr * 26))  # ~26 per year = biweekly
    print(f"  Generating {n_fake} synthetic TOAs (biweekly cadence)...")
    
    import numpy as np
    np.random.seed(42)  # or any fixed seed for a reproducible "single realization"

    fake_toas = make_fake_toas_uniform(
        startMJD=t_start,
        endMJD=t_end,
        ntoas=n_fake,
        model=m_timing,
        freq=1400 * u.MHz,
        obs='GBT',
        add_noise=add_noise,
        add_correlated_noise=False,   # ← changed
        wideband=False,
        error=noise_level_us * u.us,
    )
    print(f"  Synthetic TOAs generated. Verifying residuals are near zero...")

    # Freeze any DMX bins / JUMPs that have no matching TOAs in the synthetic set.
    # Synthetic TOAs use a single frequency/receiver, so real-data DMX bins and
    # receiver-specific JUMPs will be empty and must be frozen before fitting.
    m_timing_clean = copy.deepcopy(m_timing)
    m_timing_clean = strip_noise_model(m_timing_clean)
    m_timing_clean.find_empty_masks(fake_toas, freeze=True)

    # Verify: residuals with timing model should be ~zero
    try:
        fitter_check = WLSFitter(fake_toas, m_timing_clean)
        fitter_check.fit_toas(maxiter=3)
        resid_check = fitter_check.resids.time_resids.to(u.us).value
        print(f"  Timing model residual RMS: {np.std(resid_check):.2f} µs "
              f"(should be ~{noise_level_us:.1f} µs)")
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  WARNING: verification fit failed ({e}); continuing anyway.")
        print(f"           (The VLBI-frozen fit below is independent and may still work.)")

    # ── Step 4: Build VLBI model (freeze astrometry) ──────────────
    print(f"  Injecting VLBI astrometry and freezing...")

    # ── Diagnostic: print timing vs VLBI astrometry before injection ──
    def get_astrom(m, label):
        """Extract astrometric parameters from a model, handling both coord systems."""
        params = {}
        for p in ['RAJ','DECJ','PMRA','PMDEC','ELONG','ELAT','PMELONG','PMELAT','PX','POSEPOCH']:
            obj = getattr(m, p, None)
            if obj is not None and hasattr(obj, 'value') and obj.value is not None:
                try:
                    params[p] = float(obj.value)
                except (TypeError, ValueError):
                    params[p] = str(obj.value)
        return params

    timing_astrom = get_astrom(m_timing_clean, 'timing')

    print(f"\n  {'─'*54}")
    print(f"  ASTROMETRY DIAGNOSTIC")
    print(f"  {'─'*54}")
    print(f"  {'Parameter':<12} {'Timing (ground truth)':>22} {'VLBI (to inject)':>20} {'Delta':>12}")
    print(f"  {'─'*54}")

    # Parse VLBI injection values for display
    vlbi_display = {}
    for line in pulsar_cfg['vlbi_lines']:
        parts = line.split()
        if len(parts) >= 2 and parts[0] not in ('ECL',):
            try:
                vlbi_display[parts[0]] = float(parts[1])
            except ValueError:
                vlbi_display[parts[0]] = parts[1]

    # Also show converted ecliptic values if par uses ecliptic
    par_text = m_timing_clean.as_parfile()
    par_uses_ecl = any(l.split()[0] in ('ELONG','ELAT')
                       for l in par_text.split('\n') if l.split())
    vlbi_params_eq = [l.split()[0] for l in pulsar_cfg['vlbi_lines'] if l.strip()]
    vlbi_is_eq = any(p in ('RAJ','DECJ','PMRA','PMDEC') for p in vlbi_params_eq)
    if par_uses_ecl and vlbi_is_eq:
        converted = equatorial_to_ecliptic_vlbi(pulsar_cfg['vlbi_lines'])
        for line in converted:
            parts = line.split()
            if len(parts) >= 2 and parts[0] not in ('ECL',):
                try:
                    vlbi_display[parts[0]] = float(parts[1])
                except ValueError:
                    vlbi_display[parts[0]] = parts[1]

    all_params = sorted(set(list(timing_astrom.keys()) + list(vlbi_display.keys())))
    for p in all_params:
        t_val = timing_astrom.get(p, '—')
        v_val = vlbi_display.get(p, '—')
        try:
            delta = float(v_val) - float(t_val)
            delta_str = f"{delta:+.5f}"
        except (TypeError, ValueError):
            delta_str = '—'
        print(f"  {p:<12} {str(t_val):>22} {str(v_val):>20} {delta_str:>12}")

    print(f"  {'─'*54}\n")

    # Build VLBI model from the cleaned timing model so DMX/JUMP state is consistent
    m_vlbi = inject_vlbi_and_freeze(m_timing_clean, pulsar_cfg['vlbi_lines'])

    # Verify injection worked — print what actually ended up in the VLBI model
    vlbi_astrom = get_astrom(m_vlbi, 'vlbi')
    print(f"  POST-INJECTION VERIFICATION (values actually in VLBI model):")
    for p in all_params:
        v_before = vlbi_display.get(p, '—')
        v_after  = vlbi_astrom.get(p, '—')
        match = '✓' if str(v_before)[:6] == str(v_after)[:6] else '⚠ MISMATCH'
        print(f"    {p:<12} injected={str(v_before):>18}  in_model={str(v_after):>18}  {match}")
    print()

    # Set free parameters
    free_params = pulsar_cfg.get('free', ['F0', 'F1', 'DM', 'DM1'])
    
    # First: freeze everything in the model
    for par_name in m_vlbi.free_params:
        try:
            getattr(m_vlbi, par_name).frozen = True
        except AttributeError:
            pass
    
    # Then: free only the requested params
    for par_name in free_params:
        try:
            param = getattr(m_vlbi, par_name, None)
            if param is not None:
                param.frozen = False
        except AttributeError:
            pass

    # Freeze empty DMX/JUMP masks in VLBI model before fitting
    m_vlbi.find_empty_masks(fake_toas, freeze=True)
    print(f"  Free params: {[p for p in m_vlbi.free_params]}")


    # ── Step 5: Fit with VLBI-frozen model ────────────────────────
    print(f"  Fitting with VLBI-frozen astrometry...")
    fitter_vlbi = WLSFitter(fake_toas, m_vlbi)
    try:
        fitter_vlbi.fit_toas(maxiter=5)
    except Exception as e:
        print(f"  WARNING: Fit issue: {e}")

    residuals_us = fitter_vlbi.resids.time_resids.to(u.us).value
    # --- Manual red-noise injection (Lam-style stale-noise-model test) ---
    toa_mjds = fake_toas.get_mjds().value
    rn_us = red_noise_realization(
        toa_mjds,
        pulsar_name=pulsar_cfg['name'],
        noise_dir=NG15_NOISE_DIR,
        seed=rn_seed,
    )
    rn_rms  = float(np.std(rn_us))
    if rn_rms > 0:
        print(f"  Injected red-noise realization: RMS = {rn_rms:.2f} µs")
        residuals_us = residuals_us + rn_us
    else:
        print(f"  No PLRedNoise in timing model.")
    print(f"  Final residual RMS (fit + RN): {np.std(residuals_us):.2f} µs")
    rms = np.std(residuals_us)
    print(f"  VLBI-frozen residual RMS: {rms:.2f} µs")

    # ── Step 6: Measure annual signal ─────────────────────────────
    toa_mjds  = fake_toas.get_mjds().value
    amp, phase = fit_annual_sinusoid(toa_mjds, residuals_us)
    ls_power, ls_fap = lomb_scargle_annual(toa_mjds, residuals_us)

    # Keep times_yr for plotting only
    times_yr  = (toa_mjds - toa_mjds[0]) / 365.25

    print(f"\n  ── Annual Signal Results ──────────────────────────")
    print(f"  Simulated amplitude:  {amp:.1f} µs")
    print(f"  Paper amplitude:      {pulsar_cfg.get('paper_amplitude', '?')} µs")
    print(f"  LS power at 1yr:      {ls_power:.3f}")
    print(f"  LS FAP:               {ls_fap:.2e}")
    print(f"  Phase:                {phase:.1f}°")

    return {
        'name': name,
        'span_yr': span_yr,
        'residuals_us': residuals_us,
        'times_yr': times_yr,
        'toa_mjds': toa_mjds,
        'amp_simulated': amp,
        'amp_paper': pulsar_cfg.get('paper_amplitude', None),
        'phase': phase,
        'ls_power': ls_power,
        'ls_fap': ls_fap,
        'rms': rms,
        'm_timing': m_timing,
        'm_vlbi': m_vlbi,
    }


def plot_results(results_list):
    """
    Multi-panel figure: for each pulsar show residuals + annual fit,
    plus a final comparison plot of simulated vs paper amplitudes.
    """
    n = len(results_list)
    fig, axes = plt.subplots(n + 1, 1, figsize=(12, 3.5 * (n + 1)))
    fig.suptitle('Simulation Control: Annual Residuals from VLBI Astrometric Mismatch\n'
                 '(Synthetic TOAs with NO real annual signal — any periodicity is a fitting artifact)',
                 fontsize=12, fontweight='bold', y=1.01)

    for i, res in enumerate(results_list):
        ax = axes[i]
        t = res['times_yr']
        r = res['residuals_us']
        
        # Best-fit sinusoid
        t_fine = np.linspace(0, t.max(), 1000)
        omega  = 2 * np.pi
        A_mat = np.column_stack([
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones_like(t)
        ])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat.astype(np.float64), np.array(r, dtype=np.float64), rcond=None)
        fit_fine = (coeffs[0] * np.sin(omega * t_fine) +
                    coeffs[1] * np.cos(omega * t_fine) +
                    coeffs[2])

        ax.scatter(t, r, s=2, alpha=0.4, color='black', label='Simulated residuals')
        ax.plot(t_fine, fit_fine, 'r-', linewidth=1.5,
                label=f'Annual fit: A={res["amp_simulated"]:.1f} µs')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        
        paper_amp = res['amp_paper']
        ax.set_title(
            f'{res["name"]}  |  '
            f'Simulated: {res["amp_simulated"]:.1f} µs  |  '
            f'Paper: {paper_amp} µs  |  '
            f'LS power: {res["ls_power"]:.3f}',
            fontsize=10
        )
        ax.set_ylabel('Residual (µs)')
        ax.legend(loc='upper right', fontsize=8)
        
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (years from start)')

    # Final panel: simulated vs paper amplitudes
    ax_comp = axes[-1]
    names = [r['name'] for r in results_list]
    amp_sim   = [r['amp_simulated'] for r in results_list]
    amp_paper = [r['amp_paper'] if r['amp_paper'] is not None else 0 for r in results_list]
    
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax_comp.bar(x - width/2, amp_sim,   width, label='Simulated (no real signal)',
                         color='steelblue', alpha=0.8)
    bars2 = ax_comp.bar(x + width/2, amp_paper, width, label='Paper reported',
                         color='tomato', alpha=0.8)
    
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels(names, rotation=15, ha='right')
    ax_comp.set_ylabel('Annual Amplitude (µs)')
    ax_comp.set_title(
        'Comparison: Simulated (pure astrometric mismatch) vs Paper-Reported Amplitudes\n'
        'If bars are similar → signal is a fitting artifact',
        fontsize=10
    )
    ax_comp.legend()
    ax_comp.set_ylim(0, max(max(amp_sim or [1]), max(amp_paper or [1])) * 1.3)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax_comp.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax_comp.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*60)
    print("SIMULATION CONTROL: VLBI Astrometric Mismatch Test")
    print("="*60)
    print()
    print("Logic:")
    print("  - Generate synthetic TOAs using TIMING solution (no annual signal)")
    print("  - Inject VLBI astrometry (slightly different from timing)")
    print("  - Freeze it, refit only F0/F1/DM")
    print("  - If annual signal appears → it is a fitting artifact")
    print()

    results = []
    failed  = []

    requested = set(sys.argv[1:]) if len(sys.argv) > 1 else CLEAN_SAMPLE
    run_list  = [p for p in PULSARS if p['name'] in requested & CLEAN_SAMPLE]
    if not run_list:
        print(f"No matching pulsars. Clean sample: {sorted(CLEAN_SAMPLE)}")
        sys.exit(1)

    for cfg in run_list:
        # Check files exist
        if not os.path.exists(cfg['par']):
            print(f"\nSKIPPING {cfg['name']}: par file not found at {cfg['par']}")
            failed.append(cfg['name'])
            continue
        if not os.path.exists(cfg['tim']):
            print(f"\nSKIPPING {cfg['name']}: tim file not found at {cfg['tim']}")
            failed.append(cfg['name'])
            continue

        try:
            seed = abs(hash(cfg['name'])) & 0xFFFFFFFF
            res = run_simulation(cfg, add_noise=True, noise_level_us=1.0,
                               rn_seed=seed)
            results.append(res)
        except Exception as e:
            print(f"\nERROR on {cfg['name']}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(cfg['name'])

    if not results:
        print("\nNo results to plot. Check that your data files are accessible.")
    else:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Pulsar':<15} {'Simulated':>12} {'Paper':>10} {'Ratio':>8} {'LS Power':>10}")
        print("-" * 60)
        for res in results:
            paper_amp = res['amp_paper']
            if paper_amp:
                ratio = res['amp_simulated'] / paper_amp
                paper_str = f"{paper_amp:>8.1f}µs"
                ratio_str = f"{ratio:>8.2f}"
            else:
                paper_str = f"{'—':>10}"
                ratio_str = f"{'—':>8}"
            print(f"{res['name']:<15} {res['amp_simulated']:>10.1f}µs "
                  f"{paper_str} {ratio_str} {res['ls_power']:>10.3f}")

        print("\nGenerating plots...")
        fig = plot_results(results)
        out_path = 'vlbi_simulation_results.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {out_path}")
        plt.close()
        print("\nDone.")

