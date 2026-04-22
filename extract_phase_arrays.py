#!/usr/bin/env python3
"""
Extract year-by-year phase/amplitude arrays for Figure 3.

Run from the same directory as phase_stability_analysis.py
(i.e. the directory containing NANOGrav15yr_PulsarTiming_v2.1.0/)

Usage:
    python extract_phase_arrays.py

Output:
    Prints clean arrays for each pulsar (N_TOA >= 100 filter applied).
    Also saves phase_arrays.npz for use in figure-making script.
"""

import numpy as np

# ── Import everything from the phase stability script ─────────────────────────
# (must be in the same directory or on PYTHONPATH)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulsar_data import ( PULSARS, get_residuals, fit_sinusoid, wrap_phase_diff, mjd_to_year, MIN_TOAS )

MIN_TOAS_PAPER = 100   # stricter threshold for paper figure

results = {}

for pulsar in PULSARS:
    name = pulsar['name']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    years, res_us, err_us = get_residuals(pulsar)

    # Global fit (all TOAs)
    g = fit_sinusoid(years % 1.0, res_us, err_us)
    if g is None:
        print("  Global fit failed - skipping")
        continue
    gA, gphi, gAe, gphie = g
    print(f"  Global fit: A={gA:.2f} us  phi={gphi:.2f} deg")

    # Year-by-year fit with MIN_TOAS_PAPER threshold
    yr_min = int(np.floor(years.min()))
    yr_max = int(np.floor(years.max()))

    yr_list, A_list, phi_list, Ae_list, phie_list, n_list = [], [], [], [], [], []

    print(f"\n  {'Year':>6}  {'N_TOA':>6}  {'A(us)':>8}  {'+/-':>6}  {'phi(deg)':>9}  {'+/-':>6}  {'Include?'}")
    print(f"  {'-'*65}")

    for yr in range(yr_min, yr_max + 1):
        mask = (years >= yr) & (years < yr + 1)
        n = mask.sum()
        include = n >= MIN_TOAS_PAPER

        if n < MIN_TOAS:
            print(f"  {yr+0.5:>6.1f}  {n:>6}  {'---':>8}  {'---':>6}  {'---':>9}  {'---':>6}  SKIP (N<{MIN_TOAS})")
            continue

        result = fit_sinusoid(years[mask] % 1.0, res_us[mask], err_us[mask])
        if result is None:
            print(f"  {yr+0.5:>6.1f}  {n:>6}  {'FIT FAIL':>8}")
            continue

        A, phi, Ae, phie = result
        flag = "YES" if include else f"SPARSE (N<{MIN_TOAS_PAPER})"
        print(f"  {yr+0.5:>6.1f}  {n:>6}  {A:>8.2f}  {Ae:>6.2f}  {phi:>9.2f}  {phie:>6.2f}  {flag}")

        if include:
            yr_list.append(yr + 0.5)
            A_list.append(A);    Ae_list.append(Ae)
            phi_list.append(phi); phie_list.append(phie)
            n_list.append(n)

    if len(yr_list) < 3:
        print(f"  Too few clean years ({len(yr_list)}) - skipping")
        continue

    yr_arr   = np.array(yr_list,   dtype=np.float64)
    A_arr    = np.array(A_list,    dtype=np.float64)
    Ae_arr   = np.array(Ae_list,   dtype=np.float64)
    phi_arr  = np.array(phi_list,  dtype=np.float64)
    phie_arr = np.array(phie_list, dtype=np.float64)
    n_arr    = np.array(n_list,    dtype=np.float64)

    phi_diffs = wrap_phase_diff(phi_arr, gphi)
    phi_std   = float(np.std(phi_diffs))
    A_cv      = float(np.std(A_arr) / np.mean(A_arr))
    phi_trend = float(np.polyfit(yr_arr, phi_arr, 1)[0])

    print(f"\n  Clean years (N>={MIN_TOAS_PAPER}): {len(yr_arr)}")
    print(f"  Phase std:    {phi_std:.1f} deg")
    print(f"  Phase trend:  {phi_trend:+.2f} deg/yr")
    print(f"  Amplitude CV: {A_cv:.2f}")
    print(f"  Global A:     {gA:.2f} +/- {gAe:.2f} us")
    print(f"  Global phi:   {gphi:.2f} +/- {gphie:.2f} deg")

    results[name] = {
        'years':     yr_arr,
        'amps':      A_arr,
        'amp_errs':  Ae_arr,
        'phases':    phi_arr,
        'phase_errs': phie_arr,
        'ntoas':     n_arr,
        'gA':        gA,
        'gAe':       gAe,
        'gphi':      gphi,
        'gphie':     gphie,
        'phi_std':   phi_std,
        'phi_trend': phi_trend,
        'A_cv':      A_cv,
    }

# ── Save arrays ───────────────────────────────────────────────────────────────
if results:
    # Save as npz for figure script
    save_dict = {}
    for name, d in results.items():
        key = name.replace('+','p').replace('-','m')
        for field, val in d.items():
            save_dict[f'{key}__{field}'] = val

    np.savez('phase_arrays.npz', **save_dict)
    print(f"\n\nSaved phase_arrays.npz  ({len(results)} pulsars)")

    # Print summary table
    print(f"\n{'='*72}")
    print("SUMMARY FOR FIGURE 3")
    print(f"{'='*72}")
    print(f"{'Pulsar':<14}  {'N_yr':>5}  {'gA(us)':>8}  {'gphi(deg)':>10}  "
          f"{'phi_std':>8}  {'trend':>10}  {'A_CV':>5}")
    print(f"{'-'*72}")
    for name, d in results.items():
        print(f"{name:<14}  {len(d['years']):>5}  {d['gA']:>8.2f}  "
              f"{d['gphi']:>10.2f}  {d['phi_std']:>7.1f}°  "
              f"{d['phi_trend']:>+8.2f}°/yr  {d['A_cv']:>5.2f}")
