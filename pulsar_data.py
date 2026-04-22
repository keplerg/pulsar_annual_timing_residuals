import re
import os
import tempfile
import numpy as np
from scipy.optimize import curve_fit
import astropy.units as u
import warnings
import pint.models as pm
import pint.toa as ptoa
from pint.fitter import WLSFitter
from pint.residuals import Residuals

# ============================================================
# VLBI astrometry reference — corrected vlbi_lines for all pulsars
# Generated from source tables
#
# KEY METHODOLOGY:
#   - ELONG/ELAT from VLBI RA/Dec (NOT from NANOGrav par file)
#   - Position propagated from VLBI t_ref to par POSEPOCH using VLBI PMs
#   - NO POSEPOCH line injected — par file value is used as-is
#   - PMELONG/PMELAT from VLBI source (not timing solution)
#   - ECL IERS2010 frame throughout
#
# Sources:
#   MSPSRπ  — Ding et al. 2023, MNRAS 519:4982, Table 5
#   PSRπ    — Deller et al. 2019, ApJ 875:100, Tables 3+4 (t_ref=MJD 56000)
#   VLBA    — Chatterjee et al. 2009, ApJ 698:250, Table 4 (t_ref=MJD 52275)
# ============================================================

# ── No independent VLBI ──────────────────────────────────────────────────────
# J2234+0611 — Stovall 2018 (timing-derived only)
# J1024-0719 — Matthews 2016 (timing-derived only)

# ── Summary ──────────────────────────────────────────────────────────────────
#  Pulsar      Source        t_ref   Par POSEPOCH  Δt(yr)  drift   sep_CMB  pred
#  J0030+0451  MSPSRπ        57849      56231       -4.43   27 mas   19.6°   711 μs
#  J1012+5307  MSPSRπ        57700      56104       -4.37  112 mas   61.4°   361 μs
#  J1640+2224  MSPSRπ        57500      56246       -3.43   40 mas   85.4°    61 μs
#  J1643-1224  MSPSRπ        57700      56087       -4.42   31 mas   81.4°   113 μs
#  J1730-2304  MSPSRπ        57821      58312       +1.34   28 mas   88.6°    18 μs
#  J1738+0333  MSPSRπ        57829      57096       -2.01   17 mas   82.9°    93 μs
#  J1713+0747  Chatterjee09  52275      56232      +10.83   52 mas   88.9°    15 μs
#  J1022+1001  Deller19      56000      58042       +5.59   89 mas   21.0°   705 μs
#  J2145-0750  Deller19      56000      56105       +0.29    4 mas   26.1°   677 μs
#  J2317+1439  Deller19      56000      56213       +0.58    2 mas    7.8°   747 μs
#  J2010-1323  Deller19      56000      57019       +2.79   17 mas   49.4°   491 μs

# ── Pulsar configurations ─────────────────────────────────────────────────────
BASE = 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband'
NG15_NOISE_DIR = f'{BASE}/noise'

PULSARS = [
    {
        'name': 'J1640+2224',
        'par':  f'{BASE}/par/J1640+2224_PINT_20220305.nb.par',
        'tim':  f'{BASE}/tim/J1640+2224_PINT_20220305.nb.tim',
        # ── MSPSRπ pulsars (Ding et al. 2023) ───────────────────────────────────────
        # J1640+2224
        # Ding: RA=16:40:16.74587 Dec=+22:24:08.7642 PMRA=+2.19 PMDEC=-11.30 PX=0.68 t_ref=57500
        # Par POSEPOCH=56246  Δt=−3.43 yr  drift=39.5 mas
        'vlbi_lines': [
            "ELONG   243.9891003861047  0\n",
            "ELAT    44.0585141211592  0\n",
            "PMELONG 5.9596282741  0\n",
            "PMELAT  -10.6838196041  0\n",
            "PX      0.68  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1', 'PB', 'A1', 'TASC', 'EPS1', 'EPS2'],
        'freeze': [],
    },
    {
        'name': 'J1730-2304',
        'par':  f'{BASE}/par/J1730-2304_PINT_20220301.nb.par',
        'tim':  f'{BASE}/tim/J1730-2304_PINT_20220301.nb.tim',
        # 'par':  f'{BASE}/alternate/tempo2/J1730-2304_tempo2_20220304.par',
        # 'par':  f'{BASE}/alternate/predictive/J1730-2304_pred.par',
        # 'tim':  f'{BASE}/alternate/tim/initial/J1730-2304.Rcvr1_2.GUPPI.15y.x.nb.tim',
        # 'tim':  f'{BASE}/alternate/tim/initial/J1730-2304.Rcvr1_2.GASP.15y.x.nb.tim',
        # 'tim':  f'{BASE}/alternate/tim/initial/J1730-2304.Rcvr_800.GUPPI.15y.x.nb.tim',
        # 'tim':  f'{BASE}/alternate/tim/initial/J1730-2304.Rcvr_800.GASP.15y.x.nb.tim',
        # ── MSPSRπ pulsars (Ding et al. 2023) ───────────────────────────────────────
        # J1730-2304
        # Ding: RA=17:30:21.67969 Dec=-23:04:31.1749 PMRA=+20.3 PMDEC=-4.8 PX=1.57 t_ref=57821
        # Par POSEPOCH=58312  Δt=+1.34 yr  drift=28.0 mas
        'vlbi_lines': [
            "ELONG   263.1860837964355  0\n",
            "ELAT    0.1888616871164  0\n",
            "PMELONG 20.5196170080  0\n",
            "PMELAT  -3.7523183130  0\n",
            "PX      1.57  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1'],
        'freeze': []
    },
    {
        'name': 'J0437-4715',
        'par':  f'{BASE}/par/J0437-4715_PINT_20220301.nb.par',
        'tim':  f'{BASE}/tim/J0437-4715_PINT_20220301.nb.tim',
        # Deller et al. 2008, ApJ 685, L67
        # J0437-4715
        # RAJ = 04:37:15.99744, DECJ = −47:15:09.7170, epoch MJD 54100
        # PMRA = 121.4385 mas/yr, PMDEC = −71.4754 mas/yr
        # PX = 6.396 ± 0.054 mas (d = 156.3 ± 1.3 pc)
        'vlbi_lines': [
            "ELONG   50.4686364569  0\n",
            "ELAT    -67.8734387997  0\n",
            "PMELONG 228.611898        0\n",
            "PMELAT  -112.024039         0\n",
            "PX      6.396               0\n",
            "ECL IERS2010\n",
            # "ELONG   50.4687653835  0\n",
            # "ELAT    -67.8731936692  0\n",
            # "PMELONG 228.359205        0\n",
            # "PMELAT  -111.614346         0\n",
            # "PX      6.396               0\n",
            # "ECL IERS2010\n",
        ],
        'free': ['F0', 'F1', 'DM', 'DM1', 'PB', 'A1', 'EPS1', 'EPS2', 'T0', 'TASC', 'OM'],
        'freeze': [],
    },
    {
        'name': 'J1600-3053',
        'par':  f'{BASE}/par/J1600-3053_PINT_20220302.nb.par',
        'tim':  f'{BASE}/tim/J1600-3053_PINT_20220302.nb.tim',
        # ── EPTA Collaboration: A&A, 678, A48 (2023) ───────────────────────────────────────
        # J1600-3053
        # Par POSEPOCH=  Δt= yr  drift= mas
        'vlbi_lines': [
            "ELONG   244.3476783846  0\n",
            "ELAT    -10.0718239491  0\n",
            "PMELONG 0.472162        0\n",
            "PMELAT  -6.968467         0\n",
            "PX      0.72               0\n",
            "ECL IERS2010\n",
        ],
        'free': ['F0', 'F1', 'DM', 'DM1', 'Pb', 'PB', 'A1', 'T0', 'OM', 'ECC'],
        'freeze': [],
    },
    {
        'name': 'J1744-1134',
        'par':  f'{BASE}/par/J1744-1134_PINT_20220302.nb.par',
        'tim':  f'{BASE}/tim/J1744-1134_PINT_20220302.nb.tim',
        # ── EPTA Collaboration: A&A, 678, A48 (2023) ───────────────────────────────────────
        # J1744-1134
        # Par POSEPOCH=  Δt= yr  drift= mas
        'vlbi_lines': [
            "ELONG   266.1193969000  0\n",
            "ELAT    11.8052148400  0\n",
            "PMELONG 19.468611        0\n",
            "PMELAT  -8.865665         0\n",
            "PX      2.58               0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1'],
        'freeze': [],
    },
    {
        'name': 'J1012+5307',
        'par':  f'{BASE}/par/J1012+5307_PINT_20220305.nb.par',
        'tim':  f'{BASE}/tim/J1012+5307_PINT_20220305.nb.tim',
        # ── MSPSRπ pulsars (Ding et al. 2023) ───────────────────────────────────────
        # J1012+5307
        # Ding: RA=10:12:33.43991 Dec=+53:07:02.1110 PMRA=+2.67 PMDEC=-25.39 PX=1.17 t_ref=57700
        # Par POSEPOCH=56104  Δt=−4.37 yr  drift=111.6 mas
        'vlbi_lines': [
            "ELONG   133.3611081315788  0\n",
            "ELAT    38.7553045239115  0\n",
            "PMELONG 17.8645663989  0\n",
            "PMELAT  -21.3939457714  0\n",
            "PX      1.17  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1'],
        'freeze': [],
    },
    {
        'name': 'J1022+1001',
        'par':  f'{BASE}/par/J1022+1001_PINT_20220304.nb.par',
        'tim':  f'{BASE}/tim/J1022+1001_PINT_20220304.nb.tim',
        # ── PSRπ pulsars (Deller et al. 2019) — POSEPOCH = MJD 56000 for all ────────
        # J1022+1001
        # Deller: RA=10:22:57.9957 Dec=+10:01:52.765 PMRA=-14.921 PMDEC=+5.611 PX=1.387
        # Par POSEPOCH=58042  Δt=+5.59 yr  drift=89.1 mas
        'vlbi_lines': [
            "ELONG   153.8658325025419  0\n",
            "ELAT    -0.0639348647373  0\n",
            "PMELONG -15.9400973274  0\n",
            "PMELAT  -0.1821416367  0\n",
            "PX      1.387  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1'],
        'freeze': [],
    },
    {
        'name': 'J1713+0747',
        'par':  f'{BASE}/par/J1713+0747_PINT_20220309.nb.par',
        'tim':  f'{BASE}/tim/J1713+0747_PINT_20220309.nb.tim',
        # ── VLBA (Chatterjee et al. 2009) ────────────────────────────────────────────
        # J1713+0747
        # Chatterjee: RA=17:13:49.5306 Dec=+07:47:37.519 (NOTE: table had 37'47" typo, correct is 47'37")
        # PMRA=+4.75 PMDEC=-3.67 PX=0.95 t_ref=52275
        # Par POSEPOCH=56232  Δt=+10.83 yr  drift=51.5 mas
        'vlbi_lines': [
            "ELONG   256.6687029694823  0\n",
            "ELAT    30.7003613424412  0\n",
            "PMELONG 5.8956306213  0\n",
            "PMELAT  -3.2145087106  0\n",
            "PX      0.95  0\n",
            "ECL IERS2010\n",
        ],
        'free': ['F0', 'F1', 'DM', 'DM1', 'PB', 'PBDOT', 'A1', 'T0', 'OM'],
        'freeze': [],
    },
    {
        'name': 'J0030+0451',
        'par':  f'{BASE}/par/J0030+0451_PINT_20220302.nb.par',
        'tim':  f'{BASE}/tim/J0030+0451_PINT_20220302.nb.tim',
        # ── MSPSRπ pulsars (Ding et al. 2023) ───────────────────────────────────────
        # J0030+0451
        # Ding: RA=00:30:27.42502 Dec=+04:51:39.7159 PMRA=-6.13 PMDEC=+0.34 PX=3.02 t_ref=57849
        # Par POSEPOCH=56231  Δt=−4.43 yr  drift=27.2 mas
        'vlbi_lines': [
            "ELONG   8.9103531764166  0\n",
            "ELAT    1.4456933956831  0\n",
            "PMELONG -5.5007646283  0\n",
            "PMELAT  2.7300821813  0\n",
            "PX      3.02  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1'],
        'freeze': [],
    },
    {
        'name': 'J2317+1439',
        'par':  f'{BASE}/par/J2317+1439_PINT_20220306.nb.par',
        'tim':  f'{BASE}/tim/J2317+1439_PINT_20220306.nb.tim',
        # ── PSRπ pulsars (Deller et al. 2019) — POSEPOCH = MJD 56000 for all ────────
        # J2317+1439
        # Deller: RA=23:17:09.2364 Dec=+14:39:31.265 PMRA=-1.476 PMDEC=+3.806 PX=0.603
        # Par POSEPOCH=56213  Δt=+0.58 yr  drift=2.4 mas  (negligible)
        'vlbi_lines': [
            "ELONG   356.1294082893901  0\n",
            "ELAT    17.6802283484059  0\n",
            "PMELONG 0.2258844444  0\n",
            "PMELAT  4.0765051776  0\n",
            "PX      0.603  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1', 'PB', 'A1', 'TASC', 'EPS1', 'EPS2'],
        'freeze': [],
    },
#     {
#         'name': 'J1024-0719',
#         'par':  f'{BASE}/par/J1024-0719_PINT_20220302.nb.par',
#         'tim':  f'{BASE}/tim/J1024-0719_PINT_20220302.nb.tim',
#         # J1024-0719  [MSPSRpi T5: PMRA=-35.32 PMDEC=-48.2 PX=0.94]
#         'vlbi_lines': [
#             # Get valid ELONG/ELAT
#             "ELONG   160.734342214 0\n",
#             "ELAT    -16.044762436 0\n",
#             "PMELONG -15.028410866 0\n",
#             "PMELAT  -57.983983079 0\n",
#             "PX      0.94          0\n",
#             "ECL IERS2010\n",
#         ],
#         'free':   ['F0', 'F1', 'DM', 'DM1'],
#         'freeze': [],
#     },
    {
        'name': 'J2145-0750',
        'par':  f'{BASE}/par/J2145-0750_PINT_20220302.nb.par',
        'tim':  f'{BASE}/tim/J2145-0750_PINT_20220302.nb.tim',
        # ── PSRπ pulsars (Deller et al. 2019) — POSEPOCH = MJD 56000 for all ────────
        # J2145-0750
        # Deller: RA=21:45:50.4588 Dec=-07:50:18.514 PMRA=-9.491 PMDEC=-9.114 PX=1.603
        # Par POSEPOCH=56105  Δt=+0.29 yr  drift=3.8 mas  (negligible)
        'vlbi_lines': [
            "ELONG   326.0246121475886  0\n",
            "ELAT    5.3130479292213  0\n",
            "PMELONG -12.0358690273  0\n",
            "PMELAT  -5.4336010644  0\n",
            "PX      1.603  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1', 'PB', 'A1', 'T0', 'OM'],
        'freeze': [],
    },
#     {
#         'name': 'J2234+0611',
#         'par':  f'{BASE}/par/J2234+0611_PINT_20220304.nb.par',
#         'tim':  f'{BASE}/tim/J2234+0611_PINT_20220304.nb.tim',
#         # J2234+0611  [K.Stovall et al. 2018 T1: PMRA=25.30 PMDEC=9.31, PX=1.03]
#         'vlbi_lines': [
#             # Get valid ELONG/ELAT
#             "ELONG   342.6052307329 0\n",
#             "ELAT    14.0794370895  0\n",
#             "PMELONG 27.772236      0\n",
#             "PMELAT  -1.055194      0\n",
#             "PX      1.03           0\n",
#             "ECL IERS2010\n",
#         ],
#         'free':   ['F0', 'F1', 'DM', 'DM1'],
#         'freeze': [],
#     },
    {
        'name': 'J2010-1323',
        'par':  f'{BASE}/par/J2010-1323_PINT_20220304.nb.par',
        'tim':  f'{BASE}/tim/J2010-1323_PINT_20220304.nb.tim',
        # ── PSRπ pulsars (Deller et al. 2019) — POSEPOCH = MJD 56000 for all ────────
        # J2010-1323
        # Deller: RA=20:10:45.9211 Dec=-13:23:56.083 PMRA=+2.358 PMDEC=-5.611 PX=0.484
        # Par POSEPOCH=57019  Δt=+2.79 yr  drift=17.0 mas
        'vlbi_lines': [
            "ELONG   301.9244907795376  0\n",
            "ELAT    6.4909417624771  0\n",
            "PMELONG 1.0959697796  0\n",
            "PMELAT  -5.9881286934  0\n",
            "PX      0.484  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1'],
        'freeze': [],
    },
    {
        'name': 'J1643-1224',
        'par':  f'{BASE}/par/J1643-1224_PINT_20220305.nb.par',
        'tim':  f'{BASE}/tim/J1643-1224_PINT_20220305.nb.tim',
        # ── MSPSRπ pulsars (Ding et al. 2023) ───────────────────────────────────────
        # J1643-1224
        # Ding: RA=16:43:38.16407 Dec=-12:24:58.6531 PMRA=+6.2 PMDEC=+3.3 PX=1.31 t_ref=57700
        # Par POSEPOCH=56087  Δt=−4.42 yr  drift=31.0 mas
        'vlbi_lines': [
            "ELONG   251.0872243096196  0\n",
            "ELAT    9.7783361581640  0\n",
            "PMELONG 5.7942497365  0\n",
            "PMELAT  4.0896305866  0\n",
            "PX      1.31  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1', 'PB', 'A1', 'TASC', 'EPS1', 'EPS2'],
        'freeze': [],
    },
    {
        'name': 'J1738+0333',
        'par':  f'{BASE}/par/J1738+0333_PINT_20220302.nb.par',
        'tim':  f'{BASE}/tim/J1738+0333_PINT_20220302.nb.tim',
        # ── MSPSRπ pulsars (Ding et al. 2023) ───────────────────────────────────────
        # J1738+0333
        # Ding: RA=17:38:53.97001 Dec=+03:33:10.9124 PMRA=+6.98 PMDEC=+5.18 PX=0.50 t_ref=57829
        # Par POSEPOCH=57096  Δt=−2.01 yr  drift=17.4 mas
        'vlbi_lines': [
            "ELONG   264.0949212322083  0\n",
            "ELAT    26.8842435667497  0\n",
            "PMELONG 7.5810876343  0\n",
            "PMELAT  5.4618400184  0\n",
            "PX      0.5  0\n",
            "ECL IERS2010\n",
        ],
        'free':   ['F0', 'F1', 'DM', 'DM1', 'PB', 'A1', 'TASC', 'EPS1', 'EPS2'],
        'freeze': []
    },
]

# ── Clean sample: the 7 pulsars reported in the paper ─────────────────────────
CLEAN_SAMPLE = {
    'J0030+0451', 'J1640+2224', 'J1643-1224',
    'J1713+0747', 'J1730-2304', 'J1738+0333', 'J2317+1439',
    # 'J1012+5307',
}

# ── Strip patterns (proven from vlbi_frozen_analysis.py) ────────────────
STRIP_PATTERNS = [
    r'^FDJUMP\b', r'^DMJUMP\b', r'^JUMP\b',
    r'^EFAC\b',   r'^EQUAD\b',  r'^ECORR\b',
    r'^T2EFAC\b', r'^T2EQUAD\b',
    r'^TNEF\b',   r'^TNEQ\b',   r'^TNECORR\b',
    r'^TNRedAmp\b', r'^TNRedGam\b', r'^TNRedC\b',
    r'^RAJ\b', r'^DECJ\b',
    r'^ELONG\b', r'^ELAT\b', r'^LAMBDA\b', r'^BETA\b',
    r'^PMRA\b', r'^PMDEC\b',
    r'^PMELONG\b', r'^PMELAT\b', r'^PMLAMBDA\b', r'^PMBETA\b',
    r'^PX\b',
    r'^ECL\b',
    r'^FB0\b',
    r'^COORDS\b',
    r'^UNITS\b',
]
STRIP_RE = re.compile('|'.join(STRIP_PATTERNS), re.IGNORECASE)

MIN_TOAS = 8   # minimum TOAs per year to attempt a fit

# ── Helper functions ──────────────────────────────────────────────────────────

def mjd_to_year(mjd):
    return 2000.0 + (np.asarray(mjd) - 51544.5) / 365.25

def sinusoid(t_frac, A, phi_deg):
    return A * np.sin(2 * np.pi * t_frac + np.radians(phi_deg))

def fit_sinusoid(t_frac, res_us, err_us=None):
    if len(t_frac) < MIN_TOAS:
        l = len(t_frac)
        print(f"{l} < 8 (MIN_TOAS)")
        return None
    try:
        A0  = (np.percentile(res_us, 85) - np.percentile(res_us, 15)) / 2
        p0  = [max(A0, 1.0), 0.0]
        bnd = ([0, -360], [max(np.abs(res_us).max() * 3, 1.0), 360])
        sigma = err_us if err_us is not None else np.ones_like(res_us)
        popt, pcov = curve_fit(sinusoid, t_frac, res_us,
                               p0=p0, bounds=bnd,
                               sigma=sigma, absolute_sigma=True,
                               maxfev=20000)
        perr = np.sqrt(np.diag(pcov))
        phi  = ((popt[1] + 180) % 360) - 180
        return float(popt[0]), float(phi), float(perr[0]), float(perr[1])
    except Exception:
        print(f"fit_sinsoid exception")
        return None

def wrap_phase_diff(phi_arr, reference):
    return ((np.asarray(phi_arr) - reference + 180) % 360) - 180

def get_residuals(pulsar):
    print(f"  Loading {pulsar['name']}...")
    with open(pulsar['par']) as f:
        lines = f.readlines()

    cleaned = [l for l in lines if not STRIP_RE.match(l.strip())]
    cleaned = pulsar['vlbi_lines'] + cleaned

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False)
    tmp.writelines(cleaned)
    tmp.flush(); tmp.close()

    try:
        model = pm.get_model(tmp.name)
        toas  = ptoa.get_TOAs(pulsar['tim'], model=model,
                               ephem='DE440', planets=True)
        model.find_empty_masks(toas, freeze=True)

        for p in pulsar.get('freeze', []):
            if hasattr(model, p):
                par = getattr(model, p)
                if par.value is None:
                    par.value = 0.0
                par.frozen = True

        # Freeze everything first, then selectively free
        # This avoids the funcParameter units bug in ELL1 models (J1738)
        for par in model.params:
            getattr(model, par).frozen = True
        # Re-inject VLBI astrometry as frozen
        for par in ['RAJ','DECJ','ELONG','ELAT','PMRA','PMDEC',
                    'PMELONG','PMELAT','PX']:
            if hasattr(model, par):
                getattr(model, par).frozen = True
        # Free the requested parameters (skip funcParameters)
        from pint.models.parameter import funcParameter
        for pname in pulsar['free']:
            if hasattr(model, pname):
                p = getattr(model, pname)
                if not isinstance(p, funcParameter):
                    p.frozen = False

        fitter = WLSFitter(toas, model)
        fitter.fit_toas(maxiter=5)

        res   = Residuals(toas, fitter.model)
        mjd   = res.toas.get_mjds().value
        r_us  = res.time_resids.to(u.us).value
        e_us  = toas.get_errors().to(u.us).value
        rms   = np.sqrt(np.mean(r_us**2))
        print(f"    RMS={rms:.2f} µs  N={len(mjd)}")
        return mjd_to_year(mjd), r_us, e_us
    finally:
        os.unlink(tmp.name)

