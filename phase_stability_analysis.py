#!/usr/bin/env python3
"""
Phase stability analysis for VLBI-frozen pulsar timing residuals.

For each pulsar:
  1. Year-by-year sinusoid fit  -> amplitude and phase per year
  2. O-C diagram               -> residuals after global fit subtracted
  3. Summary statistics        -> phase std, amplitude CV

Run from the directory containing NANOGrav15yr_PulsarTiming_v2.1.0/
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

import pint.models as pm
import pint.toa as ptoa
from pint.fitter import WLSFitter
from pint.residuals import Residuals
from pulsar_data import ( PULSARS, get_residuals, sinusoid, fit_sinusoid, wrap_phase_diff, mjd_to_year, MIN_TOAS )

# ── Per-pulsar analysis ───────────────────────────────────────────────────────

def analyse(pulsar):
    name = pulsar['name']
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    years, res_us, err_us = get_residuals(pulsar)
    yr_min = int(np.floor(years.min()))
    yr_max = int(np.floor(years.max()))
    print(f"  Baseline: {years.min():.1f}-{years.max():.1f}  "
          f"({yr_max - yr_min + 1} calendar years)")

    # Global fit
    g = fit_sinusoid(years % 1.0, res_us, err_us)
    if g is None:
        print("  Global fit failed - skipping"); return None
    gA, gphi, gAe, gphie = g
    if gphi < 0:
        gphi += 360
    print(f"  Global fit: A={gA:.1f}+/-{gAe:.1f} us  phi={gphi:.1f}+/-{gphie:.1f} deg")

    # Year-by-year
    yby = {'yr': [], 'A': [], 'phi': [], 'Ae': [], 'phie': [], 'n': []}
    for yr in range(yr_min, yr_max + 1):
        mask = (years >= yr) & (years < yr + 1)
        if mask.sum() < MIN_TOAS:
            continue
        result = fit_sinusoid(years[mask] % 1.0, res_us[mask], err_us[mask])
        if result is None:
            continue
        A, phi, Ae, phie = result
        if phi < 0:
            phi += 360
        yby['yr'].append(yr + 0.5);  yby['n'].append(mask.sum())
        yby['A'].append(A);          yby['Ae'].append(Ae)
        yby['phi'].append(phi);      yby['phie'].append(phie)
    for k in yby:
        yby[k] = np.array(yby[k], dtype=np.float64)

    n_yr = len(yby['yr'])
    if n_yr < 3:
        print("  Too few years"); return None

    phi_diffs = wrap_phase_diff(yby['phi'], gphi)
    phi_std   = np.std(phi_diffs)
    A_cv      = np.std(yby['A']) / np.mean(yby['A'])
    phi_trend = float(np.polyfit(yby['yr'], yby['phi'], 1)[0])

    print(f"\n  {'Year':>6}  {'N':>5}  {'A(us)':>8}  {'+/-':>6}  {'phi(deg)':>9}  {'+/-':>6}")
    print(f"  {'-'*52}")
    for i in range(n_yr):
        print(f"  {yby['yr'][i]:>6.1f}  {yby['n'][i]:>5}  {yby['A'][i]:>8.1f}  "
              f"{yby['Ae'][i]:>6.1f}  {yby['phi'][i]:>9.1f}  {yby['phie'][i]:>6.1f}")

    stable_phi = phi_std < 20
    stable_A   = A_cv   < 0.4
    print(f"\n  Phase std:    {phi_std:.1f} deg  "
          f"({'highly stable' if stable_phi else 'moderate' if phi_std < 45 else 'drifting'})")
    print(f"  Phase trend:  {phi_trend:+.2f} deg/yr")
    print(f"  Amplitude CV: {A_cv:.2f}  "
          f"({'stable' if stable_A else 'moderate' if A_cv < 0.6 else 'variable'})")

    # O-C
    global_signal = sinusoid(years % 1.0, gA, gphi)
    oc = res_us - global_signal
    oc_yr, oc_med, oc_err = [], [], []
    for yr in range(yr_min, yr_max + 1):
        mask = (years >= yr) & (years < yr + 1)
        if mask.sum() < MIN_TOAS:
            continue
        oc_yr.append(yr + 0.5)
        oc_med.append(np.median(oc[mask]))
        oc_err.append(np.std(oc[mask]) / np.sqrt(mask.sum()))
    oc_yr  = np.array(oc_yr,  dtype=np.float64)
    oc_med = np.array(oc_med, dtype=np.float64)
    oc_err = np.array(oc_err, dtype=np.float64)

    # Plot
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f"{name}  -  Phase Stability Analysis\n"
        f"Global fit: A = {gA:.1f} us,  phi = {gphi:.1f} deg  |  "
        f"Phase sigma = {phi_std:.1f} deg  |  Amplitude CV = {A_cv:.2f}",
        fontsize=11
    )
    gs_fig = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # Full residuals
    ax1 = fig.add_subplot(gs_fig[0, :])
    ax1.scatter(years, res_us, s=1.5, alpha=0.25, color='steelblue', rasterized=True)
    t_d = np.linspace(years.min(), years.max(), 3000)
    ax1.plot(t_d, sinusoid(t_d % 1.0, gA, gphi),
             'r-', lw=1.8, label=f'Global fit  A={gA:.1f} us  phi={gphi:.1f} deg')
    ax1.axhline(0, color='k', lw=0.5)
    ax1.set_xlabel('Year'); ax1.set_ylabel('Residual (us)')
    ax1.set_title('Full residuals with global annual fit')
    ax1.legend(fontsize=9)

    # Year-by-year amplitude
    ax2 = fig.add_subplot(gs_fig[1, 0])
    ax2.errorbar(yby['yr'], yby['A'], yerr=yby['Ae'],
                 fmt='o', color='steelblue', capsize=4, ms=6, lw=1.5)
    ax2.axhline(gA, color='r', lw=1.2, ls='--', label=f'Global A={gA:.1f} us')
    ax2.fill_between([yby['yr'].min()-0.6, yby['yr'].max()+0.6],
                     [gA - gAe]*2, [gA + gAe]*2, alpha=0.12, color='r')
    ax2.set_xlabel('Year'); ax2.set_ylabel('Amplitude (us)')
    ax2.set_title(f'Year-by-year amplitude  (CV = {A_cv:.2f})')
    ax2.legend(fontsize=8)
    ax2.set_xlim(yby['yr'].min()-0.6, yby['yr'].max()+0.6)

    # Year-by-year phase
    ax3 = fig.add_subplot(gs_fig[1, 1])
    ax3.errorbar(yby['yr'], yby['phi'], yerr=yby['phie'],
                 fmt='s', color='darkorange', capsize=4, ms=6, lw=1.5)
    ax3.axhline(gphi, color='r', lw=1.2, ls='--', label=f'Global phi={gphi:.1f} deg')
    phi_m = np.mean(yby['phi'])
    ax3.fill_between([yby['yr'].min()-0.6, yby['yr'].max()+0.6],
                     [phi_m - phi_std]*2, [phi_m + phi_std]*2,
                     alpha=0.15, color='green', label=f'+/-{phi_std:.1f} deg')
    t_fit = np.linspace(yby['yr'].min(), yby['yr'].max(), 200)
    ax3.plot(t_fit, np.polyval(np.polyfit(yby['yr'], yby['phi'], 1), t_fit),
             'b--', lw=1, label=f'Trend {phi_trend:+.1f} deg/yr')
    ax3.set_xlabel('Year'); ax3.set_ylabel('Phase (deg)')
    ax3.set_title(f'Year-by-year phase  (sigma = {phi_std:.1f} deg)')
    ax3.legend(fontsize=7)
    ax3.set_xlim(yby['yr'].min()-0.6, yby['yr'].max()+0.6)

    # O-C
    ax4 = fig.add_subplot(gs_fig[2, 0])
    ax4.errorbar(oc_yr, oc_med, yerr=oc_err,
                 fmt='D', color='purple', capsize=4, ms=6, lw=1.5)
    ax4.axhline(0, color='r', lw=1.2, ls='--')
    if len(oc_yr) > 2:
        zoc = np.polyfit(oc_yr, oc_med, 1)
        ax4.plot(t_fit, np.polyval(zoc, t_fit), 'b--', lw=1,
                 label=f'Trend {zoc[0]:+.1f} us/yr')
        ax4.legend(fontsize=8)
    ax4.set_xlabel('Year'); ax4.set_ylabel('O-C (us)')
    ax4.set_title('O-C: residuals after global fit subtracted')
    ax4.set_xlim(oc_yr.min()-0.6, oc_yr.max()+0.6)

    # Phase histogram
    ax5 = fig.add_subplot(gs_fig[2, 1])
    nbins = min(10, max(5, n_yr // 2))
    ax5.hist(yby['phi'], bins=nbins, color='darkorange',
             alpha=0.75, edgecolor='black', lw=0.7)
    ax5.axvline(gphi, color='r',     lw=1.5, ls='--',
                label=f'Global phi={gphi:.1f} deg')
    ax5.axvline(phi_m - phi_std, color='green', lw=1.0, ls=':')
    ax5.axvline(phi_m + phi_std, color='green', lw=1.0, ls=':',
                label=f'sigma={phi_std:.1f} deg')
    ax5.set_xlabel('Phase (deg)'); ax5.set_ylabel('Count')
    ax5.set_title('Phase distribution across years')
    ax5.legend(fontsize=8)

    outfile = f"{name.replace('+','p')}_phase_stability.png"
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")

    return {
        'name':      name,    'n_yr':    n_yr,
        'gA':        gA,      'gAe':     gAe,
        'gphi':      gphi,    'gphie':   gphie,
        'A_mean':    float(np.mean(yby['A'])),
        'A_cv':      A_cv,    'phi_std': phi_std,
        'phi_trend': phi_trend,
    }

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

    if results:
        print(f"\n{'='*72}")
        print('PHASE STABILITY SUMMARY')
        print(f"{'='*72}")
        print(f"  {'Pulsar':<14} {'N_yr':>5} {'A_mean':>8} {'A_CV':>6} "
              f"{'phi_std':>8} {'phi_trend':>11}  Assessment")
        print(f"  {'-'*72}")
        for r in results:
            stable_phi = r['phi_std'] < 20
            stable_A   = r['A_cv']   < 0.4
            assess = ('coherent signal'   if stable_phi and stable_A else
                      'partially stable'  if stable_phi or  stable_A else
                      'likely systematic')
            print(f"  {r['name']:<14} {r['n_yr']:>5} {r['A_mean']:>8.1f} "
                  f"{r['A_cv']:>6.2f} {r['phi_std']:>7.1f} deg "
                  f"{r['phi_trend']:>+8.1f} deg/yr  {assess}")
        print()
        print('  phi_std < 20 deg  = highly phase-stable over full baseline')
        print('  phi_trend ~ 0     = phase not drifting (rules out growing systematic)')
        print('  A_CV < 0.4        = amplitude stable year to year')
