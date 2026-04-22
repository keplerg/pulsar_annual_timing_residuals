"""
J1713+0747 Annual Residual Analysis
Frozen VLBI astrometry from Chatterjee et al. 2009 (ApJ 698, 250)

Strategy:
- Strip FB0 from par file (degenerate with PB, causes WLS crash)
- Strip noise/jump parameters that cause PINT parser errors
- Inject VLBI astrometry and freeze it
- Free only: F0, F1, DM*, binary orbital params (PB-based, not FB0)
"""

import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import astropy.units as u
from astropy.time import Time
from astropy.timeseries import LombScargle
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic, Galactic

from pint.models import get_model_and_toas
import pint.models as pm
import pint.toa as toa
from pint.fitter import WLSFitter

from pulsar_data import ( PULSARS, STRIP_RE, get_residuals, fit_sinusoid, wrap_phase_diff, mjd_to_year, MIN_TOAS )

# CMB dipole direction (Planck 2018)
CMB_DIPOLE_GAL_L = 264.021  # degrees
CMB_DIPOLE_GAL_B = 48.253   # degrees
CMB_DIPOLE_VELOCITY = 369.82  # km/s
EARTH_ORBITAL_VELOCITY = 29.78  # km/s

def analyse(pulsar):
    name = pulsar['name']
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    # ── 1. Clean par file ─────────────────────────────────────────────────────────

    with open(pulsar['par']) as f:
        lines = f.readlines()

    cleaned = [l for l in lines if not STRIP_RE.match(l.strip())]

    cleaned = pulsar['vlbi_lines'] + cleaned

    tmp_par = tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False)
    tmp_par.writelines(cleaned)
    tmp_par.flush()
    tmp_par_name = tmp_par.name
    tmp_par.close()
    print(f"Cleaned par written to: {tmp_par_name}")

    # DEBUG: show all astrometry lines that made it through
    print("=== Astrometry in cleaned par ===")
    with open(tmp_par_name) as f:
        for line in f:
            if any(line.upper().startswith(k) for k in 
                ['RAJ','DECJ','ELONG','ELAT','LAMBDA','BETA','PMEL','PMRA','PMDEC','PX']):
                print(" ", line.rstrip())
    print("=================================")

    # ── 2. Load ───────────────────────────────────────────────────────────────────
    print("Loading model and TOAs...")
    m, t = get_model_and_toas(tmp_par_name, pulsar['tim'], allow_T2=True, allow_name_mixing=True)
    print(f"  {len(t)} TOAs  MJD {t.get_mjds().min():.1f} – {t.get_mjds().max():.1f}")


    # ── 3. Parameter setup ────────────────────────────────────────────────────────
    from pint.models.parameter import funcParameter

    # Freeze everything
    for par in m.params:
        getattr(m, par).frozen = True

    # Free spin + DM — but only if NOT a funcParameter
    def safe_free(model, par):
        if hasattr(model, par):
            p = getattr(model, par)
            if not isinstance(p, funcParameter):
                p.frozen = False
            else:
                print(f"  Skipping funcParameter: {par}")

    for par in pulsar['free']:
        safe_free(m, par)

    # Ensure astrometry and FB0 stay frozen regardless
    for par in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX', 'FB0', 'SINI', 'M2']:
        if hasattr(m, par):
            getattr(m, par).frozen = True

    free = [p for p in m.params if not getattr(m, p).frozen]
    print(f"\nFree ({len(free)}): {', '.join(free)}")

    # ── 4. Fit ────────────────────────────────────────────────────────────────────
    print("\nFitting...")
    fitter = WLSFitter(t, m)
    # Guard against None-valued parameters in free list
    for par in list(free):
        p = getattr(fitter.model, par, None)
        if p is not None and p.value is None:
            p.frozen = True
            free.remove(par)
            print(f"  Frozen {par} (value is None)")
    fitter.fit_toas()
    try:
        print(fitter.get_summary())
    except Exception as e:
        print(f"(Summary skipped: {e})")
        for par in free:
            try:
                p = getattr(fitter.model, par)
                print(f"  {par} = {p.value}")
            except AttributeError:
                pass

    # ── 5. Residuals ──────────────────────────────────────────────────────────────
    residuals_us = fitter.resids.time_resids.to(u.us).value.astype(float)
    mjds         = t.get_mjds().value
    times        = Time(mjds, format='mjd').datetime
    rms_total    = np.std(residuals_us)
    print(f"\nRMS: {rms_total:.2f} µs  ({len(residuals_us)} TOAs)")


    # ── 6. Lomb-Scargle ───────────────────────────────────────────────────────────
    frequency, power = LombScargle(mjds, residuals_us).autopower(
        minimum_frequency=1.0 / (5.0 * 365.25),
        maximum_frequency=1.0 / (0.4 * 365.25),
    )
    period_years = (1.0 / frequency) / 365.25
    peak_power   = power.max()
    peak_period  = period_years[np.argmax(power)]
    fap          = LombScargle(mjds, residuals_us).false_alarm_probability(peak_power)
    annual_power = LombScargle(mjds, residuals_us).power(1.0 / 365.25)

    print(f"\nLS peak: {peak_period:.3f} yr  power={peak_power:.4f}  FAP={fap:.2e}")
    print(f"Power at exactly 1 yr: {annual_power:.4f}")

    # ── 7. Annual sinusoid fit ────────────────────────────────────────────────────
    omega  = 2.0 * np.pi / 365.25
    # Use fractional year from Jan 1 to match phase_stability convention
    year_frac = np.array([Time(m, format='mjd').to_value('decimalyear') % 1.0 for m in mjds])
    phase = 2 * np.pi * year_frac
    # phase  = omega * mjds
    A      = np.column_stack([np.sin(phase), np.cos(phase), np.ones_like(phase)])
    # coeffs, _, _, _ = np.linalg.lstsq(A, residuals_us, rcond=None)
    # Replace the current unweighted lstsq with:
    errs_us = t.get_errors().to(u.us).value.astype(float)
    w = 1.0 / errs_us**2

    Aw = np.column_stack([np.sin(phase), np.cos(phase), np.ones_like(phase)]) * np.sqrt(w)[:,None]
    bw = residuals_us * np.sqrt(w)
    coeffs, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    sin_amp, cos_amp, offset = coeffs
    amplitude = np.sqrt(sin_amp**2 + cos_amp**2)
    phase_deg = np.degrees(np.arctan2(cos_amp, sin_amp))
    if phase_deg < 0:
        phase_deg += 360
    fitted    = A @ coeffs
    rms_after = np.std(residuals_us - fitted)

    print(f"\nAnnual fit:  A = {amplitude:.2f} µs  φ = {phase_deg:+.1f}°  "
          f"RMS {rms_total:.2f} → {rms_after:.2f} µs  "
          f"({(1-(rms_after/rms_total)**2)*100:.1f}% variance removed)")

    # ── 8. Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(
        f"{name}  —  VLBI-frozen astrometry\n",
        fontsize=11
    )
    outfile = f"{name}_vlbi_frozen_residuals.png"
    print(outfile)

    ax = axes[0]
    ax.scatter(times, residuals_us, s=2, alpha=0.5, color='steelblue')
    si = np.argsort(mjds)
    ax.plot(np.array(times)[si], fitted[si], 'r-', lw=1.5,
            label=f'Annual fit: {amplitude:.1f} µs pk,  φ={phase_deg:+.0f}°')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Date"); ax.set_ylabel("Residual (µs)")
    ax.set_title("Timing Residuals (frozen VLBI astrometry)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    ax = axes[1]
    # uses same Jan 1 = 0 convention as the fit
    pf = np.array([Time(m, format='mjd').to_value('decimalyear') % 1.0 for m in mjds])
    th = np.linspace(0, 1, 500)
    ff = sin_amp*np.sin(2*np.pi*th) + cos_amp*np.cos(2*np.pi*th) + offset
    ax.scatter(pf, residuals_us, s=2, alpha=0.4, color='steelblue')
    ax.plot(th, ff, 'r-', lw=1.5, label=f'A={amplitude:.1f} µs  φ={phase_deg:+.0f}°')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Phase (0–1 = 1 year)"); ax.set_ylabel("Residual (µs)")
    ax.set_title("Residuals Folded at 1 Year")
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(period_years, power, color='steelblue', lw=0.8)
    ax.axvline(1.0, color='r', lw=1.5, ls='--', label='1 year')
    ax.axvline(0.5, color='orange', lw=1.0, ls=':', label='6 months')
    ax.set_xlabel("Period (years)"); ax.set_ylabel("LS Power")
    ax.set_title(f"Lomb-Scargle  peak={peak_period:.3f} yr  FAP={fap:.1e}")
    ax.legend(fontsize=9)
    ax.set_xlim(0.3, 5.0)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    # plt.show()
    print(f"\nSaved: {outfile}")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = []
    for p in PULSARS:
        try:
            r = analyse(p)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR {p['name']}: {e}")
            import traceback; traceback.print_exc()

