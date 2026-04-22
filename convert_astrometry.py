#!/usr/bin/env python3
"""
convert_astrometry.py
=====================
Convert pulsar astrometric parameters between equatorial and ecliptic
coordinate systems, as used in PINT par files.

Usage examples:
  # Equatorial → Ecliptic
  python3 convert_astrometry.py eq2ecl \
      --raj "17:38:53.97001" --decj "+03:33:10.9124" \
      --pmra 6.98 --pmdec 5.18 --px 0.50

  # Ecliptic → Equatorial
  python3 convert_astrometry.py ecl2eq \
      --elong 264.0949201067 --elat 26.8842557191 \
      --pmelong 7.8214 --pmelat 4.7911 --px 0.50

  # From par file — position from par, optionally override PM/PX with paper values
  python3 convert_astrometry.py frompar J1713+0747.par
  python3 convert_astrometry.py frompar J1713+0747.par --pmra 4.917 --pmdec -3.905 --px 0.95
  python3 convert_astrometry.py frompar J1713+0747.par --pmelong 7.8214 --pmelat 4.7911 --px 0.50

  # Interactive mode (no arguments)
  python3 convert_astrometry.py
"""

import numpy as np
import sys
import argparse

# Obliquity of the ecliptic (IAU 1980, matches PINT's IERS2010 convention)
EPS_DEG = 23.4392911
EPS_RAD = np.radians(EPS_DEG)


# ── Parsing helpers ───────────────────────────────────────────────────────────

def parse_raj(s):
    """Parse RAJ string 'HH:MM:SS.sss' → degrees."""
    s = s.strip()
    parts = s.split(':')
    h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
    return 15.0 * (h + m/60.0 + sec/3600.0)


def parse_decj(s):
    """Parse DECJ string '±DD:MM:SS.sss' → degrees."""
    s = s.strip()
    sign = -1.0 if s.startswith('-') else 1.0
    s = s.lstrip('+-')
    parts = s.split(':')
    d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
    return sign * (d + m/60.0 + sec/3600.0)


def deg_to_raj(deg):
    """Degrees → 'HH:MM:SS.sssssssss' string."""
    h_total = deg / 15.0
    h  = int(h_total)
    m_total = (h_total - h) * 60.0
    m  = int(m_total)
    s  = (m_total - m) * 60.0
    return f"{h:02d}:{m:02d}:{s:013.10f}"


def deg_to_decj(deg):
    """Degrees → '±DD:MM:SS.ssssssss' string."""
    sign = '+' if deg >= 0 else '-'
    deg  = abs(deg)
    d    = int(deg)
    m_total = (deg - d) * 60.0
    m    = int(m_total)
    s    = (m_total - m) * 60.0
    return f"{sign}{d:02d}:{m:02d}:{s:012.9f}"


# ── Par file reader ───────────────────────────────────────────────────────────

def read_par_astrometry(par_path):
    """
    Read astrometric parameters from a PINT-format .par file.

    Handles both equatorial (RAJ/DECJ/PMRA/PMDEC) and ecliptic
    (ELONG/ELAT/PMELONG/PMELAT) coordinate systems, as well as the
    alternate LAMBDA/BETA names used by some par files.

    Returns a dict with keys:
      mode        : 'equatorial' or 'ecliptic'
      ra_deg      : float or None   (equatorial only)
      dec_deg     : float or None   (equatorial only)
      elon_deg    : float or None   (ecliptic only)
      elat_deg    : float or None   (ecliptic only)
      pmra        : float or None
      pmdec       : float or None
      pmelong     : float or None
      pmelat      : float or None
      px          : float or None
      raw         : dict of raw string values keyed by parameter name
    """
    raw = {}
    with open(par_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].upper()
            val = parts[1]
            raw[key] = val

    result = {
        'mode': None,
        'ra_deg': None, 'dec_deg': None,
        'elon_deg': None, 'elat_deg': None,
        'pmra': None, 'pmdec': None,
        'pmelong': None, 'pmelat': None,
        'px': None,
        'raw': raw,
    }

    # Position — equatorial
    if 'RAJ' in raw:
        result['ra_deg']  = parse_raj(raw['RAJ'])
        result['mode']    = 'equatorial'
    if 'DECJ' in raw:
        result['dec_deg'] = parse_decj(raw['DECJ'])
        result['mode']    = 'equatorial'

    # Position — ecliptic (ELONG/ELAT or LAMBDA/BETA)
    for elong_key in ('ELONG', 'LAMBDA'):
        if elong_key in raw:
            result['elon_deg'] = float(raw[elong_key])
            result['mode']     = 'ecliptic'
            break
    for elat_key in ('ELAT', 'BETA'):
        if elat_key in raw:
            result['elat_deg'] = float(raw[elat_key])
            result['mode']     = 'ecliptic'
            break

    # Proper motions — equatorial
    if 'PMRA'  in raw: result['pmra']    = float(raw['PMRA'])
    if 'PMDEC' in raw: result['pmdec']   = float(raw['PMDEC'])

    # Proper motions — ecliptic
    if 'PMELONG' in raw: result['pmelong'] = float(raw['PMELONG'])
    if 'PMELAT'  in raw: result['pmelat']  = float(raw['PMELAT'])

    # Parallax
    if 'PX' in raw: result['px'] = float(raw['PX'])

    return result


def print_par_summary(par_data, par_path, overrides):
    """Print a clear summary of what came from the par file vs overrides."""
    print(f"\n  Par file : {par_path}")
    print(f"  Mode     : {par_data['mode']}")

    def src(key):
        return "** OVERRIDE **" if key in overrides else "(from par)"

    if par_data['mode'] == 'equatorial':
        raj  = deg_to_raj(par_data['ra_deg'])
        decj = deg_to_decj(par_data['dec_deg'])
        print(f"  RAJ      = {raj}   (from par)")
        print(f"  DECJ     = {decj}  (from par)")
        if par_data['pmra']  is not None: print(f"  PMRA     = {par_data['pmra']}   {src('pmra')}")
        if par_data['pmdec'] is not None: print(f"  PMDEC    = {par_data['pmdec']}  {src('pmdec')}")
    else:
        print(f"  ELONG    = {par_data['elon_deg']:.10f}  (from par)")
        print(f"  ELAT     = {par_data['elat_deg']:.10f}  (from par)")
        if par_data['pmelong'] is not None: print(f"  PMELONG  = {par_data['pmelong']}  {src('pmelong')}")
        if par_data['pmelat']  is not None: print(f"  PMELAT   = {par_data['pmelat']}   {src('pmelat')}")

    if par_data['px'] is not None:
        print(f"  PX       = {par_data['px']}  {src('px')}")


def run_frompar(par_path,
                pmra=None, pmdec=None, px_override=None,
                pmelong=None, pmelat=None,
                posepoch_in=None, posepoch_out=None):
    """
    Read position from par file; optionally override PM and/or PX
    with values from an independent source (e.g. Chatterjee, Ding).
    Then convert in whichever direction the par file's coordinate
    system implies (equatorial→ecliptic or ecliptic→equatorial).
    """
    par_data = read_par_astrometry(par_path)

    if par_data['mode'] is None:
        print(f"ERROR: Could not determine coordinate system in {par_path}")
        print("  Expected RAJ/DECJ or ELONG/ELAT (or LAMBDA/BETA).")
        sys.exit(1)

    # Track which values were overridden for display purposes
    overrides = set()

    if par_data['mode'] == 'equatorial':
        # Apply any overrides
        if pmra    is not None: par_data['pmra']  = pmra;    overrides.add('pmra')
        if pmdec   is not None: par_data['pmdec'] = pmdec;   overrides.add('pmdec')
        if px_override is not None: par_data['px'] = px_override; overrides.add('px')

        # Check we have enough to convert
        missing = [k for k in ('pmra', 'pmdec') if par_data[k] is None]
        if missing:
            print(f"ERROR: Par file has no {'/'.join(missing).upper()} and none was supplied.")
            print("  Use --pmra / --pmdec to provide values from your paper.")
            sys.exit(1)

        print("\n" + "="*60)
        print("FROM PAR FILE  (equatorial → ecliptic)")
        print("="*60)
        print_par_summary(par_data, par_path, overrides)
        print_full_result_eq2ecl(par_data['ra_deg'], par_data['dec_deg'],
                                  par_data['pmra'],   par_data['pmdec'],
                                  par_data['px'],
                                  posepoch_in=posepoch_in,
                                  posepoch_out=posepoch_out)

    else:  # ecliptic
        # Apply any overrides — accept either ecliptic or equatorial PM overrides
        if pmelong is not None: par_data['pmelong'] = pmelong; overrides.add('pmelong')
        if pmelat  is not None: par_data['pmelat']  = pmelat;  overrides.add('pmelat')
        if px_override is not None: par_data['px']  = px_override; overrides.add('px')

        # If user supplied equatorial PM overrides, convert them to ecliptic first
        if pmra is not None or pmdec is not None:
            if pmra is None or pmdec is None:
                print("ERROR: Supply both --pmra and --pmdec together, or neither.")
                sys.exit(1)
            # Convert equatorial PM to ecliptic using the par file position
            _, _, cvt_pmelong, cvt_pmelat = eq_to_ecl(
                *ecl_to_eq(par_data['elon_deg'], par_data['elat_deg'], 0.0, 0.0)[:2],
                pmra, pmdec
            )
            par_data['pmelong'] = cvt_pmelong
            par_data['pmelat']  = cvt_pmelat
            overrides.update({'pmelong', 'pmelat'})
            print(f"\n  Note: --pmra/--pmdec converted to ecliptic PM "
                  f"({cvt_pmelong:.6f}, {cvt_pmelat:.6f} mas/yr)")

        missing = [k for k in ('pmelong', 'pmelat') if par_data[k] is None]
        if missing:
            print(f"ERROR: Par file has no {'/'.join(missing).upper()} and none was supplied.")
            print("  Use --pmelong / --pmelat (or --pmra / --pmdec) to provide values.")
            sys.exit(1)

        print("\n" + "="*60)
        print("FROM PAR FILE  (ecliptic → equatorial)")
        print("="*60)
        print_par_summary(par_data, par_path, overrides)
        print_full_result_ecl2eq(par_data['elon_deg'], par_data['elat_deg'],
                                  par_data['pmelong'],  par_data['pmelat'],
                                  par_data['px'])



def eq_to_ecl(ra_deg, dec_deg, pmra, pmdec):
    """
    Equatorial → Ecliptic conversion.

    The coordinate rotation is Rx(-eps) around the x-axis (vernal equinox).
    The PM vector is built in equatorial Cartesian, then rotated by the same
    Rx(-eps) matrix into ecliptic Cartesian before projecting onto the
    ecliptic tangent-plane basis vectors.  Mixing equatorial pm_vec with
    ecliptic basis vectors (the old unit-vector approach) gives wrong answers
    because the basis vectors are expressed in different Cartesian frames.

    Parameters
    ----------
    ra_deg  : float  Right ascension in degrees
    dec_deg : float  Declination in degrees
    pmra    : float  Proper motion in RA (μα·cosδ) in mas/yr
    pmdec   : float  Proper motion in Dec in mas/yr

    Returns
    -------
    elon_deg, elat_deg, pmelong, pmelat  (all floats, degrees or mas/yr)
    """
    ra_r  = np.radians(ra_deg)
    dec_r = np.radians(dec_deg)

    # Position
    sin_elat = (np.sin(dec_r)*np.cos(EPS_RAD)
                - np.cos(dec_r)*np.sin(EPS_RAD)*np.sin(ra_r))
    elat_r   = np.arcsin(np.clip(sin_elat, -1.0, 1.0))
    sin_elon = (np.sin(dec_r)*np.sin(EPS_RAD)
                + np.cos(dec_r)*np.cos(EPS_RAD)*np.sin(ra_r))
    cos_elon = np.cos(dec_r) * np.cos(ra_r)
    elon_r   = np.arctan2(sin_elon, cos_elon) % (2*np.pi)

    # PM: equatorial Cartesian tangent vector
    p_eq = np.array([-np.sin(ra_r),  np.cos(ra_r), 0.0])
    q_eq = np.array([-np.sin(dec_r)*np.cos(ra_r),
                     -np.sin(dec_r)*np.sin(ra_r),
                      np.cos(dec_r)])
    pm_eq_cart = pmra * p_eq + pmdec * q_eq

    # Rotate into ecliptic Cartesian via Rx(-eps):
    #   x' = x
    #   y' = y*cos(eps) + z*sin(eps)
    #   z' = -y*sin(eps) + z*cos(eps)
    pm_ecl_cart = np.array([
        pm_eq_cart[0],
        pm_eq_cart[1]*np.cos(EPS_RAD) + pm_eq_cart[2]*np.sin(EPS_RAD),
       -pm_eq_cart[1]*np.sin(EPS_RAD) + pm_eq_cart[2]*np.cos(EPS_RAD)
    ])

    # Project onto ecliptic tangent-plane basis (in ecliptic Cartesian)
    p_ecl = np.array([-np.sin(elon_r),  np.cos(elon_r), 0.0])
    q_ecl = np.array([-np.sin(elat_r)*np.cos(elon_r),
                      -np.sin(elat_r)*np.sin(elon_r),
                       np.cos(elat_r)])
    pmelong = np.dot(pm_ecl_cart, p_ecl) / np.cos(elat_r)
    pmelat  = np.dot(pm_ecl_cart, q_ecl)

    return np.degrees(elon_r), np.degrees(elat_r), pmelong, pmelat


def ecl_to_eq(elon_deg, elat_deg, pmelong, pmelat):
    """
    Ecliptic → Equatorial conversion.

    Inverse of eq_to_ecl: rotate PM vector from ecliptic Cartesian back to
    equatorial Cartesian using Rx(+eps) (inverse of Rx(-eps)).

    Parameters
    ----------
    elon_deg : float  Ecliptic longitude in degrees
    elat_deg : float  Ecliptic latitude in degrees
    pmelong  : float  Proper motion in ecliptic longitude (μλ, NOT ·cosβ) in mas/yr
    pmelat   : float  Proper motion in ecliptic latitude in mas/yr

    Returns
    -------
    ra_deg, dec_deg, pmra, pmdec  (all floats, degrees or mas/yr)
    """
    elon_r = np.radians(elon_deg)
    elat_r = np.radians(elat_deg)

    # Position: inverse rotation Rx(+eps)
    sin_dec = (np.sin(elat_r)*np.cos(EPS_RAD)
               + np.cos(elat_r)*np.sin(EPS_RAD)*np.sin(elon_r))
    dec_r   = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    y_eq    = (-np.sin(elat_r)*np.sin(EPS_RAD)
               + np.cos(elat_r)*np.cos(EPS_RAD)*np.sin(elon_r))
    x_eq    = np.cos(elat_r)*np.cos(elon_r)
    ra_r    = np.arctan2(y_eq, x_eq) % (2*np.pi)

    # PM: ecliptic Cartesian tangent vector
    p_ecl = np.array([-np.sin(elon_r),  np.cos(elon_r), 0.0])
    q_ecl = np.array([-np.sin(elat_r)*np.cos(elon_r),
                      -np.sin(elat_r)*np.sin(elon_r),
                       np.cos(elat_r)])
    pm_ecl_cart = pmelong * np.cos(elat_r) * p_ecl + pmelat * q_ecl

    # Rotate into equatorial Cartesian via Rx(+eps):
    #   x' = x
    #   y' = y*cos(eps) - z*sin(eps)
    #   z' = y*sin(eps) + z*cos(eps)
    pm_eq_cart = np.array([
        pm_ecl_cart[0],
        pm_ecl_cart[1]*np.cos(EPS_RAD) - pm_ecl_cart[2]*np.sin(EPS_RAD),
        pm_ecl_cart[1]*np.sin(EPS_RAD) + pm_ecl_cart[2]*np.cos(EPS_RAD)
    ])

    # Project onto equatorial tangent-plane basis (in equatorial Cartesian)
    p_eq = np.array([-np.sin(ra_r),  np.cos(ra_r), 0.0])
    q_eq = np.array([-np.sin(dec_r)*np.cos(ra_r),
                     -np.sin(dec_r)*np.sin(ra_r),
                      np.cos(dec_r)])
    pmra  = np.dot(pm_eq_cart, p_eq)
    pmdec = np.dot(pm_eq_cart, q_eq)

    return np.degrees(ra_r), np.degrees(dec_r), pmra, pmdec


# ── Epoch propagation ─────────────────────────────────────────────────────────

def propagate_position(ra_deg, dec_deg, pmra, pmdec, t_in, t_out):
    """
    Propagate an equatorial position from epoch t_in to epoch t_out (both MJD).

    Uses the small-angle linear approximation appropriate for pulsar astrometry
    over baselines of order decades.  The proper motion components follow the
    standard pulsar-timing convention:
        pmra  = μα · cos(δ)   [mas/yr]   — what PINT calls PMRA
        pmdec = μδ            [mas/yr]   — what PINT calls PMDEC

    Parameters
    ----------
    ra_deg, dec_deg : float   Position at t_in (degrees)
    pmra, pmdec     : float   Proper motion (mas/yr, μα·cosδ convention)
    t_in, t_out     : float   MJD epochs

    Returns
    -------
    ra_out, dec_out : float   Position at t_out (degrees)
    """
    dt_yr = (t_out - t_in) / 365.25

    # ΔRA in degrees: pmra is already μα·cosδ so divide by cosδ to get true
    # angular shift, then divide by 3600*1000 to convert mas→degrees.
    cos_dec = np.cos(np.radians(dec_deg))
    d_ra_deg  = (pmra  * dt_yr) / (cos_dec * 3600.0 * 1000.0)
    d_dec_deg = (pmdec * dt_yr) / (          3600.0 * 1000.0)

    return ra_deg + d_ra_deg, dec_deg + d_dec_deg


# ── Output formatters ─────────────────────────────────────────────────────────

def print_ecl_parlines(elon, elat, pmelong, pmelat, px, source=''):
    print()
    if source:
        print(f"  # Source: {source}")
    print(f'  "ELONG   {elon:.10f}  0\\n",')
    print(f'  "ELAT    {elat:.10f}  0\\n",')
    print(f'  "PMELONG {pmelong:.6f}        0\\n",')
    print(f'  "PMELAT  {pmelat:.6f}         0\\n",')
    if px is not None:
        print(f'  "PX      {px}               0\\n",')
    print(f'  "ECL IERS2010\\n",')


def print_eq_parlines(ra_deg, dec_deg, pmra, pmdec, px, source=''):
    raj  = deg_to_raj(ra_deg)
    decj = deg_to_decj(dec_deg)
    print()
    if source:
        print(f"  # Source: {source}")
    print(f'  "RAJ     {raj}  0\\n",')
    print(f'  "DECJ    {decj} 0\\n",')
    print(f'  "PMRA    {pmra:.6f}        0\\n",')
    print(f'  "PMDEC   {pmdec:.6f}         0\\n",')
    if px is not None:
        print(f'  "PX      {px}               0\\n",')


def print_full_result_eq2ecl(ra_deg, dec_deg, pmra, pmdec, px,
                              posepoch_in=None, posepoch_out=None):
    """
    Convert equatorial → ecliptic, optionally propagating position first.

    If posepoch_in and posepoch_out are both supplied, the position is
    propagated from posepoch_in to posepoch_out before the rotation.
    The proper motion is unchanged by propagation (linear model).
    """
    ra_in, dec_in = ra_deg, dec_deg  # keep originals for display

    propagated = (posepoch_in is not None and posepoch_out is not None
                  and posepoch_in != posepoch_out)

    if propagated:
        ra_deg, dec_deg = propagate_position(
            ra_deg, dec_deg, pmra, pmdec, posepoch_in, posepoch_out)

    elon, elat, pmelong, pmelat = eq_to_ecl(ra_deg, dec_deg, pmra, pmdec)
    raj_in  = deg_to_raj(ra_in)
    decj_in = deg_to_decj(dec_in)
    raj_out  = deg_to_raj(ra_deg)
    decj_out = deg_to_decj(dec_deg)

    print("\n" + "="*60)
    print("EQUATORIAL → ECLIPTIC")
    print("="*60)

    if propagated:
        dt_yr = (posepoch_out - posepoch_in) / 365.25
        d_ra_mas  = (ra_deg  - ra_in)  * 3600 * 1000
        d_dec_mas = (dec_deg - dec_in) * 3600 * 1000
        print(f"\nEpoch propagation: MJD {posepoch_in} → MJD {posepoch_out}"
              f"  (Δt = {dt_yr:.4f} yr)")
        print(f"  Input  position : RAJ={raj_in}  DECJ={decj_in}")
        print(f"  Propagated      : RAJ={raj_out} DECJ={decj_out}")
        print(f"  ΔRA = {d_ra_mas:+.3f} mas    ΔDec = {d_dec_mas:+.3f} mas")

    print(f"\nInput (equatorial, MJD {posepoch_out if propagated else 'as supplied'}):")
    print(f"  RAJ     = {raj_out}")
    print(f"  DECJ    = {decj_out}")
    print(f"  RA      = {ra_deg:.10f} deg")
    print(f"  Dec     = {dec_deg:.10f} deg")
    print(f"  PMRA    = {pmra} mas/yr  (μα·cosδ)")
    print(f"  PMDEC   = {pmdec} mas/yr")
    if px is not None:
        print(f"  PX      = {px} mas")

    print(f"\nOutput (ecliptic):")
    print(f"  ELONG   = {elon:.10f} deg")
    print(f"  ELAT    = {elat:.10f} deg")
    print(f"  PMELONG = {pmelong:.6f} mas/yr  (μλ, not ·cosβ)")
    print(f"  PMELAT  = {pmelat:.6f} mas/yr")
    if px is not None:
        print(f"  PX      = {px} mas  (unchanged)")

    print(f"\nPINT vlbi_lines (ecliptic):")
    print_ecl_parlines(elon, elat, pmelong, pmelat, px)

    # Round-trip check (from propagated position)
    ra2, dec2, pmra2, pmdec2 = ecl_to_eq(elon, elat, pmelong, pmelat)
    print(f"\nRound-trip check (ecl→eq):")
    print(f"  ΔRA   = {(ra2-ra_deg)*3600*1000:.4f} mas")
    print(f"  ΔDec  = {(dec2-dec_deg)*3600*1000:.4f} mas")
    print(f"  ΔPMRA = {pmra2-pmra:.6f} mas/yr")
    print(f"  ΔPMDec= {pmdec2-pmdec:.6f} mas/yr")


def print_full_result_ecl2eq(elon, elat, pmelong, pmelat, px):
    ra_deg, dec_deg, pmra, pmdec = ecl_to_eq(elon, elat, pmelong, pmelat)
    raj  = deg_to_raj(ra_deg)
    decj = deg_to_decj(dec_deg)

    print("\n" + "="*60)
    print("ECLIPTIC → EQUATORIAL")
    print("="*60)
    print(f"\nInput (ecliptic):")
    print(f"  ELONG   = {elon} deg")
    print(f"  ELAT    = {elat} deg")
    print(f"  PMELONG = {pmelong} mas/yr  (μλ, not ·cosβ)")
    print(f"  PMELAT  = {pmelat} mas/yr")
    if px is not None:
        print(f"  PX      = {px} mas")

    print(f"\nOutput (equatorial):")
    print(f"  RAJ     = {raj}")
    print(f"  DECJ    = {decj}")
    print(f"  RA      = {ra_deg:.10f} deg")
    print(f"  Dec     = {dec_deg:.10f} deg")
    print(f"  PMRA    = {pmra:.6f} mas/yr  (μα·cosδ)")
    print(f"  PMDEC   = {pmdec:.6f} mas/yr")
    if px is not None:
        print(f"  PX      = {px} mas  (unchanged)")

    print(f"\nPINT vlbi_lines (equatorial):")
    print_eq_parlines(ra_deg, dec_deg, pmra, pmdec, px)

    # Round-trip check
    elon2, elat2, pe2, pb2 = eq_to_ecl(ra_deg, dec_deg, pmra, pmdec)
    print(f"\nRound-trip check (eq→ecl):")
    print(f"  ΔELONG  = {(elon2-elon)*3600*1000:.4f} mas")
    print(f"  ΔELAT   = {(elat2-elat)*3600*1000:.4f} mas")
    print(f"  ΔPMELONG= {pe2-pmelong:.6f} mas/yr")
    print(f"  ΔPMELAT = {pb2-pmelat:.6f} mas/yr")


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive():
    print("\nAstrometry Coordinate Converter")
    print("================================")
    print("1) Equatorial → Ecliptic")
    print("2) Ecliptic   → Equatorial")
    print("3) From par file (position from par, optionally override PM/PX)")
    choice = input("\nSelect (1/2/3): ").strip()

    if choice == '1':
        raj    = input("RAJ   (HH:MM:SS.sss)  : ").strip()
        decj   = input("DECJ  (±DD:MM:SS.sss) : ").strip()
        pmra   = float(input("PMRA  (mas/yr)        : "))
        pmdec  = float(input("PMDEC (mas/yr)        : "))
        px_s   = input("PX    (mas, or blank) : ").strip()
        px     = float(px_s) if px_s else None
        ra_deg  = parse_raj(raj)
        dec_deg = parse_decj(decj)
        print_full_result_eq2ecl(ra_deg, dec_deg, pmra, pmdec, px)

    elif choice == '2':
        elon    = float(input("ELONG   (deg)         : "))
        elat    = float(input("ELAT    (deg)         : "))
        pmelong = float(input("PMELONG (mas/yr)      : "))
        pmelat  = float(input("PMELAT  (mas/yr)      : "))
        px_s    = input("PX      (mas, or blank): ").strip()
        px      = float(px_s) if px_s else None
        print_full_result_ecl2eq(elon, elat, pmelong, pmelat, px)

    elif choice == '3':
        par_path = input("Par file path         : ").strip()
        print("\nOverride PM/PX from paper? (leave blank to use par file values)")
        par_data = read_par_astrometry(par_path)

        pmra = pmdec = pmelong = pmelat = px_ov = None

        if par_data['mode'] == 'equatorial':
            s = input("PMRA  (mas/yr, blank=keep par): ").strip()
            if s: pmra  = float(s)
            s = input("PMDEC (mas/yr, blank=keep par): ").strip()
            if s: pmdec = float(s)
        else:
            print("Par file uses ecliptic coords. Override with:")
            print("  a) Ecliptic PM  b) Equatorial PM  c) Keep par values")
            pm_choice = input("Select (a/b/c): ").strip().lower()
            if pm_choice == 'a':
                s = input("PMELONG (mas/yr): ").strip()
                if s: pmelong = float(s)
                s = input("PMELAT  (mas/yr): ").strip()
                if s: pmelat  = float(s)
            elif pm_choice == 'b':
                pmra  = float(input("PMRA  (mas/yr): "))
                pmdec = float(input("PMDEC (mas/yr): "))

        s = input("PX (mas, blank=keep par)      : ").strip()
        if s: px_ov = float(s)

        run_frompar(par_path, pmra=pmra, pmdec=pmdec, px_override=px_ov,
                    pmelong=pmelong, pmelat=pmelat)
    else:
        print("Invalid choice.")



# ── CLI mode ──────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) == 1:
        interactive()
        return

    parser = argparse.ArgumentParser(description='Pulsar astrometry coordinate converter')
    sub = parser.add_subparsers(dest='cmd')

    p1 = sub.add_parser('eq2ecl', help='Equatorial → Ecliptic')
    p1.add_argument('--raj',   required=True, help='RAJ  HH:MM:SS.sss')
    p1.add_argument('--decj',  required=True, help='DECJ ±DD:MM:SS.sss')
    p1.add_argument('--pmra',  type=float, required=True, help='PMRA  mas/yr (μα·cosδ)')
    p1.add_argument('--pmdec', type=float, required=True, help='PMDEC mas/yr')
    p1.add_argument('--px',    type=float, default=None,  help='PX mas (optional)')
    p1.add_argument('--posepoch_in',  type=float, default=None,
                    help='VLBI reference epoch (MJD) — propagate FROM this epoch')
    p1.add_argument('--posepoch_out', type=float, default=None,
                    help='Par file POSEPOCH (MJD)   — propagate TO this epoch')

    p2 = sub.add_parser('ecl2eq', help='Ecliptic → Equatorial')
    p2.add_argument('--elong',   type=float, required=True)
    p2.add_argument('--elat',    type=float, required=True)
    p2.add_argument('--pmelong', type=float, required=True, help='PMELONG mas/yr (not ·cosβ)')
    p2.add_argument('--pmelat',  type=float, required=True)
    p2.add_argument('--px',      type=float, default=None)

    p3 = sub.add_parser('frompar',
        help='Read position from par file; optionally override PM/PX with paper values')
    p3.add_argument('parfile', help='Path to PINT .par file')
    p3.add_argument('--pmra',    type=float, default=None,
                    help='Override PMRA (mas/yr, μα·cosδ) — equatorial')
    p3.add_argument('--pmdec',   type=float, default=None,
                    help='Override PMDEC (mas/yr) — equatorial')
    p3.add_argument('--pmelong', type=float, default=None,
                    help='Override PMELONG (mas/yr, not ·cosβ) — ecliptic')
    p3.add_argument('--pmelat',  type=float, default=None,
                    help='Override PMELAT (mas/yr) — ecliptic')
    p3.add_argument('--px',      type=float, default=None,
                    help='Override PX (mas)')
    p3.add_argument('--posepoch_in',  type=float, default=None,
                    help='VLBI reference epoch (MJD) — propagate FROM this epoch')
    p3.add_argument('--posepoch_out', type=float, default=None,
                    help='Par file POSEPOCH (MJD)   — propagate TO this epoch')

    args = parser.parse_args()

    if args.cmd == 'eq2ecl':
        ra_deg  = parse_raj(args.raj)
        dec_deg = parse_decj(args.decj)
        print_full_result_eq2ecl(ra_deg, dec_deg, args.pmra, args.pmdec, args.px,
                                  posepoch_in=args.posepoch_in,
                                  posepoch_out=args.posepoch_out)
    elif args.cmd == 'ecl2eq':
        print_full_result_ecl2eq(args.elong, args.elat, args.pmelong, args.pmelat, args.px)
    elif args.cmd == 'frompar':
        run_frompar(args.parfile,
                    pmra=args.pmra, pmdec=args.pmdec, px_override=args.px,
                    pmelong=args.pmelong, pmelat=args.pmelat,
                    posepoch_in=args.posepoch_in,
                    posepoch_out=args.posepoch_out)
    else:
        parser.print_help()


# ── Built-in self-test ────────────────────────────────────────────────────────

def run_builtin_examples():
    """Print the two pulsars you asked about plus a round-trip test."""
    print("\n" + "="*60)
    print("BUILT-IN EXAMPLES")
    print("="*60)

    examples = [
        dict(name="J1738+0333 (MSPSRpi Table 5)",
             raj="17:38:53.97001", decj="+03:33:10.9124",
             pmra=6.98, pmdec=5.18, px=0.50),
        dict(name="J1713+0747 (Chatterjee 2009)",
             raj="17:13:49.53027", decj="+07:47:37.4900",
             pmra=4.917, pmdec=-3.905, px=0.95),
    ]

    for ex in examples:
        print(f"\n--- {ex['name']} ---")
        ra  = parse_raj(ex['raj'])
        dec = parse_decj(ex['decj'])
        print_full_result_eq2ecl(ra, dec, ex['pmra'], ex['pmdec'], ex['px'])


if __name__ == '__main__':
    if '--examples' in sys.argv:
        run_builtin_examples()
    else:
        main()
