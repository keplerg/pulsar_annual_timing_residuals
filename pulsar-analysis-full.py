"""
Pulsar Annual Signal Analysis
Data loading, basic fitting, and geometric calculations.

This script analyzes timing residuals for annual signals and compares
their properties to predictions from position errors vs CMB-frame effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic, Galactic
import pint.models as pm
import pint.toa as toa
from pint.residuals import Residuals
from astropy.coordinates import get_sun
import warnings
warnings.filterwarnings('ignore')
"""
Pulsar Annual Signal Analysis - Part 1
Data loading, basic fitting, and geometric calculations.

This script analyzes timing residuals for annual signals and compares
their properties to predictions from position errors vs CMB-frame effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic, Galactic
import pint.models as pm
import pint.toa as toa
from pint.residuals import Residuals
import warnings
warnings.filterwarnings('ignore')

# CMB dipole direction (Planck 2018)
CMB_DIPOLE_GAL_L = 264.021  # degrees
CMB_DIPOLE_GAL_B = 48.253   # degrees
CMB_DIPOLE_VELOCITY = 369.82  # km/s
EARTH_ORBITAL_VELOCITY = 29.78  # km/s

def get_cmb_dipole_ecliptic():
    """Get CMB dipole direction in ecliptic coordinates."""
    cmb_gal = SkyCoord(l=CMB_DIPOLE_GAL_L*u.deg, b=CMB_DIPOLE_GAL_B*u.deg, frame='galactic')
    return cmb_gal.transform_to(BarycentricMeanEcliptic())

def load_pulsar_data(par_file, tim_file):
    """Load TOAs and timing model."""
    print(f"Loading {par_file}...")
    toas_obj = toa.get_TOAs(tim_file, ephem="DE440", planets=True, include_bipm=True)
    model = pm.get_model(par_file)
    model.find_empty_masks(toas_obj, freeze=True)
    return toas_obj, model

def get_pulsar_coordinates(model):
    """Extract pulsar coordinates from model, handling both equatorial and ecliptic."""
    if hasattr(model, 'RAJ'):
        psr_coord = SkyCoord(ra=model.RAJ.quantity, dec=model.DECJ.quantity, frame='icrs')
    elif hasattr(model, 'ELONG'):
        ecl_lon = model.ELONG.quantity
        ecl_lat = model.ELAT.quantity
        psr_coord = SkyCoord(lon=ecl_lon, lat=ecl_lat, frame=BarycentricMeanEcliptic()).icrs
    else:
        raise ValueError("Cannot find position parameters in model")
    
    psr_ecl = psr_coord.transform_to(BarycentricMeanEcliptic())
    psr_gal = psr_coord.transform_to(Galactic)
    
    cmb_dipole = SkyCoord(l=CMB_DIPOLE_GAL_L*u.deg, b=CMB_DIPOLE_GAL_B*u.deg, frame='galactic')
    cmb_angle = psr_coord.separation(cmb_dipole).deg
    
    return {
        'ra_deg': psr_coord.ra.deg,
        'dec_deg': psr_coord.dec.deg,
        'ecl_lon_deg': psr_ecl.lon.deg,
        'ecl_lat_deg': psr_ecl.lat.deg,
        'gal_l_deg': psr_gal.l.deg,
        'gal_b_deg': psr_gal.b.deg,
        'cmb_angle_deg': cmb_angle,
        'coord_icrs': psr_coord,
        'coord_ecl': psr_ecl
    }

def fit_annual_signal(mjds, residuals, include_semiannual=True):
    """Fit annual (and optionally semi-annual) sinusoid to residuals."""
    # Convert to float64 to avoid float128 issues with scipy
    t_days = np.asarray(mjds - mjds[0], dtype=np.float64)
    residuals_f64 = np.asarray(residuals, dtype=np.float64)
    mjd0 = float(mjds[0])
    
    omega = 2 * np.pi / 365.25
    
    if include_semiannual:
        def model_func(t, offset, slope, A1, phi1, A2, phi2):
            return (offset + slope * t + 
                    A1 * np.sin(omega * t + phi1) + 
                    A2 * np.sin(2 * omega * t + phi2))
        p0 = [np.mean(residuals_f64), 0, np.std(residuals_f64)*0.1, 0, np.std(residuals_f64)*0.01, 0]
    else:
        def model_func(t, offset, slope, A1, phi1):
            return offset + slope * t + A1 * np.sin(omega * t + phi1)
        p0 = [np.mean(residuals_f64), 0, np.std(residuals_f64)*0.1, 0]
    
    try:
        popt, pcov = curve_fit(model_func, t_days, residuals_f64, p0=p0, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        
        if include_semiannual:
            return {
                'offset': popt[0], 'offset_err': perr[0],
                'slope': popt[1], 'slope_err': perr[1],
                'A1': popt[2], 'A1_err': perr[2],
                'phi1': popt[3], 'phi1_err': perr[3],
                'A2': popt[4], 'A2_err': perr[4],
                'phi2': popt[5], 'phi2_err': perr[5],
                'model_func': model_func,
                'popt': popt,
                't_days': t_days,
                'mjd0': mjd0
            }
        else:
            return {
                'offset': popt[0], 'offset_err': perr[0],
                'slope': popt[1], 'slope_err': perr[1],
                'A1': popt[2], 'A1_err': perr[2],
                'phi1': popt[3], 'phi1_err': perr[3],
                'model_func': model_func,
                'popt': popt,
                't_days': t_days,
                'mjd0': mjd0
            }
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None

def phase_to_earth_longitude(phi_deg):
    """
    Convert fitted phase (fractional-year convention, t=0 = Jan 1)
    to day-of-year of signal maximum and Earth ecliptic longitude.
    
    Signal: A*sin(2*pi*t + phi), t in fractional years from Jan 1
    Maximum when 2*pi*t + phi = pi/2
    => t_max = (pi/2 - phi_rad) / (2*pi)  [fractional year]
    """
    phi_rad = np.radians(phi_deg)
    t_max_frac = (np.pi/2 - phi_rad) / (2 * np.pi)
    # Wrap to [0, 1)
    t_max_frac = t_max_frac % 1.0
    doy = t_max_frac * 365.25
    
    # Earth ecliptic longitude: vernal equinox ~March 20 (DOY ~79)
    # At vernal equinox Earth is at lon=180 (Sun at lon=0)
    # Earth moves ~1 deg/day
    earth_lon = (180 + (doy - 79) * 360/365.25) % 360
    
    return doy, earth_lon

def expected_phase_position_error(psr_ecl_lon):
    """
    Expected Earth longitude at maximum residual for a position error.
    For RA error: max when Earth velocity is perpendicular to pulsar direction.
    """
    return (psr_ecl_lon + 90) % 360

def analyze_single_pulsar(par_file, tim_file, pulsar_name=None):
    """Complete analysis for a single pulsar."""
    
    # Load data
    toas_obj, model = load_pulsar_data(par_file, tim_file)
    
    if pulsar_name is None:
        pulsar_name = str(model.PSR.value) if hasattr(model, 'PSR') else "Unknown"
    
    # Get coordinates
    coords = get_pulsar_coordinates(model)
    
    # Get pre-fit residuals
    prefit_res = Residuals(toas_obj, model)
    
    # Convert everything to float64 to avoid float128 issues
    R_prefit = np.asarray(prefit_res.time_resids.to(u.s).value, dtype=np.float64)
    mjds = np.asarray(toas_obj.get_mjds().value, dtype=np.float64)
    freqs = np.asarray(toas_obj.get_freqs().value, dtype=np.float64)
    
    # Convert MJDs to years for plotting
    start_year = Time(mjds[0], format='mjd').decimalyear
    years = (mjds - mjds[0]) / 365.25 + start_year
    
    # Fit annual signal
    fit_result = fit_annual_signal(mjds, R_prefit, include_semiannual=True)
    
    if fit_result is None:
        return None
    
    # Compute phase diagnostics
    earth_lon_at_max, mjd_at_max = phase_to_earth_longitude(fit_result['phi1']) # , fit_result['mjd0'])
    expected_lon_pos_err = expected_phase_position_error(coords['ecl_lon_deg'])
    phase_diff = ((earth_lon_at_max - expected_lon_pos_err + 180) % 360) - 180
    
    results = {
        'pulsar_name': pulsar_name,
        'n_toas': len(mjds),
        'mjd_range': (mjds.min(), mjds.max()),
        'year_range': (years.min(), years.max()),
        'freq_range_mhz': (freqs.min(), freqs.max()),
        'coordinates': coords,
        'fit': fit_result,
        'prefit_rms_us': np.std(R_prefit) * 1e6,
        'A1_us': abs(fit_result['A1']) * 1e6,
        'A1_err_us': fit_result['A1_err'] * 1e6,
        'A2_us': abs(fit_result['A2']) * 1e6,
        'A2_err_us': fit_result['A2_err'] * 1e6,
        'earth_lon_at_max': earth_lon_at_max,
        'expected_lon_pos_err': expected_lon_pos_err,
        'phase_diff_deg': phase_diff,
        'mjds': mjds,
        'years': years,
        'freqs': freqs,
        'residuals': R_prefit
    }
    
    return results

def print_analysis_summary(results):
    """Print summary of analysis results."""
    print("\n" + "="*70)
    print(f"PULSAR: {results['pulsar_name']}")
    print("="*70)
    
    print(f"\nData Summary:")
    print(f"  TOAs: {results['n_toas']}")
    print(f"  Time span: {results['year_range'][0]:.1f} - {results['year_range'][1]:.1f}")
    print(f"  Freq range: {results['freq_range_mhz'][0]:.0f} - {results['freq_range_mhz'][1]:.0f} MHz")
    print(f"  Pre-fit RMS: {results['prefit_rms_us']:.1f} μs")
    
    coords = results['coordinates']
    print(f"\nCoordinates:")
    print(f"  RA, Dec: {coords['ra_deg']:.3f}°, {coords['dec_deg']:.3f}°")
    print(f"  Ecl lon, lat: {coords['ecl_lon_deg']:.1f}°, {coords['ecl_lat_deg']:.1f}°")
    print(f"  Angle from CMB dipole: {coords['cmb_angle_deg']:.1f}°")
    
    print(f"\nAnnual Signal Fit:")
    print(f"  Annual amplitude: {results['A1_us']:.1f} ± {results['A1_err_us']:.1f} μs")
    print(f"  Semi-annual amplitude: {results['A2_us']:.1f} ± {results['A2_err_us']:.1f} μs")
    print(f"  Signal significance: {results['A1_us']/results['A1_err_us']:.1f} σ")
    
    print(f"\nPhase Analysis:")
    print(f"  Earth longitude at max residual: {results['earth_lon_at_max']:.1f}°")
    print(f"  Expected for position error: {results['expected_lon_pos_err']:.1f}°")
    print(f"  Phase difference: {results['phase_diff_deg']:.1f}°")
    
    if abs(results['phase_diff_deg']) < 30:
        print("  → Phase CONSISTENT with position error")
    elif abs(results['phase_diff_deg']) > 60:
        print("  → Phase INCONSISTENT with position error")
    else:
        print("  → Phase inconclusive")

def plot_single_pulsar_analysis(results, save_prefix=None):
    """Create diagnostic plots for a single pulsar."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    years = results['years']
    R_us = results['residuals'] * 1e6
    fit = results['fit']
    t_days = fit['t_days']
    
    # Panel 1: Raw pre-fit residuals
    axes[0].plot(years, R_us, 'k.', markersize=1, alpha=0.3)
    axes[0].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Pre-fit Residual (μs)')
    axes[0].set_title(f"{results['pulsar_name']} - Pre-fit Timing Residuals")
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Detrended with fit overlay
    linear_trend = fit['offset'] + fit['slope'] * t_days
    R_detrended = (results['residuals'] - linear_trend) * 1e6
    
    axes[1].plot(years, R_detrended, 'k.', markersize=1, alpha=0.3)
    
    # Model curve
    t_model = np.linspace(t_days.min(), t_days.max(), 1000)
    omega = 2 * np.pi / 365.25
    sinusoid = (fit['A1'] * np.sin(omega * t_model + fit['phi1']) + 
                fit['A2'] * np.sin(2 * omega * t_model + fit['phi2'])) * 1e6
    years_model = t_model / 365.25 + results['year_range'][0]
    
    axes[1].plot(years_model, sinusoid, 'r-', linewidth=2,
                 label=f"Annual: {results['A1_us']:.0f} μs, Semi-annual: {results['A2_us']:.0f} μs")
    axes[1].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Detrended Residual (μs)')
    axes[1].set_title('After Removing Linear Trend')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Final residuals after removing fit
    full_model = fit['model_func'](t_days, *fit['popt'])
    R_final = (results['residuals'] - full_model) * 1e6
    
    axes[2].plot(years, R_final, 'k.', markersize=1, alpha=0.3)
    axes[2].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Final Residual (μs)')
    axes[2].set_xlabel('Year')
    axes[2].set_title('After Removing Linear Trend + Sinusoids')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_residuals.png', dpi=150)
    
    return fig

def analyze_frequency_dependence(results, freq_bands=None):
    """Check if annual signal varies with frequency."""
    
    if freq_bands is None:
        fmin, fmax = results['freq_range_mhz']
        if fmax - fmin > 1000:
            freq_bands = [(fmin, fmin+500), (fmin+500, fmin+1500), (fmin+1500, fmax)]
        else:
            freq_bands = [(fmin, (fmin+fmax)/2), ((fmin+fmax)/2, fmax)]
    
    freqs = results['freqs']
    mjds = results['mjds']
    residuals = results['residuals']
    
    omega = 2 * np.pi / 365.25
    t_days = mjds - mjds[0]
    
    def annual_model(t, offset, slope, A1, phi1):
        return offset + slope * t + A1 * np.sin(omega * t + phi1)
    
    freq_results = []
    
    for f_low, f_high in freq_bands:
        mask = (freqs >= f_low) & (freqs < f_high)
        n_toas = mask.sum()
        
        if n_toas < 50:
            continue
        
        try:
            p0 = [0, 0, 1e-5, 0]
            popt, pcov = curve_fit(annual_model, t_days[mask], residuals[mask], p0=p0)
            perr = np.sqrt(np.diag(pcov))
            
            freq_results.append({
                'freq_band': f'{f_low:.0f}-{f_high:.0f} MHz',
                'freq_center': (f_low + f_high) / 2,
                'n_toas': n_toas,
                'A1_us': abs(popt[2]) * 1e6,
                'A1_err_us': perr[2] * 1e6,
                'phi1': popt[3],
                'phi1_err': perr[3]
            })
        except Exception as e:
            print(f"  Fit failed for {f_low:.0f}-{f_high:.0f} MHz: {e}")
    
    return freq_results

def plot_frequency_analysis(results, freq_results, save_prefix=None):
    """Plot frequency dependence of annual signal."""
    
    if len(freq_results) < 2:
        print("Not enough frequency bands for comparison")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    freqs = [r['freq_center'] for r in freq_results]
    amps = [r['A1_us'] for r in freq_results]
    amp_errs = [r['A1_err_us'] for r in freq_results]
    
    # Amplitude vs frequency
    axes[0].errorbar(freqs, amps, yerr=amp_errs, fmt='o-', capsize=5, markersize=8)
    axes[0].set_xlabel('Frequency (MHz)')
    axes[0].set_ylabel('Annual Amplitude (μs)')
    axes[0].set_title(f"{results['pulsar_name']}: Frequency Dependence")
    axes[0].grid(True, alpha=0.3)
    
    # Add ν^-2 expectation for ISM
    if len(freqs) >= 2:
        f_ref = freqs[0]
        a_ref = amps[0]
        f_model = np.linspace(min(freqs)*0.9, max(freqs)*1.1, 100)
        a_model_ism = a_ref * (f_ref / f_model) ** 2
        axes[0].plot(f_model, a_model_ism, 'r--', alpha=0.5, label='ν⁻² (ISM)')
        axes[0].legend()
    
    # Phase vs frequency
    phases = [r['phi1'] for r in freq_results]
    phase_errs = [r['phi1_err'] for r in freq_results]
    
    axes[1].errorbar(freqs, phases, yerr=phase_errs, fmt='o-', capsize=5, markersize=8)
    axes[1].set_xlabel('Frequency (MHz)')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_title('Phase vs Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_freq_analysis.png', dpi=150)
    
    return fig

def compute_solar_elongation(results):
    """Compute solar elongation for each TOA."""
    
    psr_coord = results['coordinates']['coord_icrs']
    times = Time(results['mjds'], format='mjd')
    sun_positions = get_sun(times)
    elongations = psr_coord.separation(sun_positions).deg
    
    return elongations

def plot_solar_elongation(results, save_prefix=None):
    """Check for correlation with solar elongation."""
    
    elongations = compute_solar_elongation(results)
    R_us = results['residuals'] * 1e6
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    axes[0].scatter(elongations, R_us, alpha=0.1, s=1)
    axes[0].set_xlabel('Solar Elongation (deg)')
    axes[0].set_ylabel('Residual (μs)')
    axes[0].set_title(f"{results['pulsar_name']}: Residuals vs Solar Elongation")
    axes[0].grid(True, alpha=0.3)
    
    # Binned average
    bins = np.linspace(0, 180, 19)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    
    for i in range(len(bins)-1):
        mask = (elongations >= bins[i]) & (elongations < bins[i+1])
        if mask.sum() > 10:
            bin_means.append(np.mean(R_us[mask]))
            bin_stds.append(np.std(R_us[mask]) / np.sqrt(mask.sum()))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    axes[1].errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=3)
    axes[1].axhline(0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Solar Elongation (deg)')
    axes[1].set_ylabel('Mean Residual (μs)')
    axes[1].set_title('Binned Average')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_solar_elong.png', dpi=150)
    
    return fig

def compare_multiple_pulsars(all_results, save_prefix=None):
    """Compare annual signals across multiple pulsars."""
    
    if len(all_results) < 2:
        print("Need at least 2 pulsars for comparison")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    names = [r['pulsar_name'] for r in all_results]
    ecl_lats = [r['coordinates']['ecl_lat_deg'] for r in all_results]
    cmb_angles = [r['coordinates']['cmb_angle_deg'] for r in all_results]
    amplitudes = [r['A1_us'] for r in all_results]
    amp_errs = [r['A1_err_us'] for r in all_results]
    phase_diffs = [r['phase_diff_deg'] for r in all_results]
    
    # Amplitude vs ecliptic latitude
    axes[0,0].errorbar(ecl_lats, amplitudes, yerr=amp_errs, fmt='o', capsize=5, markersize=8)
    for i, name in enumerate(names):
        axes[0,0].annotate(name, (ecl_lats[i], amplitudes[i]), fontsize=8,
                          xytext=(5, 5), textcoords='offset points')
    
    # Add cos(ecl_lat) curve for position error expectation
    lat_range = np.linspace(-90, 90, 100)
    if max(amplitudes) > 0:
        scale = max(amplitudes) / max(abs(np.cos(np.radians(ecl_lats))))
        axes[0,0].plot(lat_range, scale * abs(np.cos(np.radians(lat_range))), 
                      'r--', alpha=0.5, label='∝ |cos(β)|')
        axes[0,0].legend()
    
    axes[0,0].set_xlabel('Ecliptic Latitude (deg)')
    axes[0,0].set_ylabel('Annual Amplitude (μs)')
    axes[0,0].set_title('Position Error Signature')
    axes[0,0].grid(True, alpha=0.3)
    
    # Amplitude vs CMB angle
    axes[0,1].errorbar(cmb_angles, amplitudes, yerr=amp_errs, fmt='o', capsize=5, markersize=8)
    for i, name in enumerate(names):
        axes[0,1].annotate(name, (cmb_angles[i], amplitudes[i]), fontsize=8,
                          xytext=(5, 5), textcoords='offset points')
    axes[0,1].set_xlabel('Angle from CMB Dipole (deg)')
    axes[0,1].set_ylabel('Annual Amplitude (μs)')
    axes[0,1].set_title('CMB Frame Signature')
    axes[0,1].grid(True, alpha=0.3)
    
    # Phase difference from position error expectation
    axes[1,0].bar(range(len(names)), phase_diffs)
    axes[1,0].set_xticks(range(len(names)))
    axes[1,0].set_xticklabels(names, rotation=45, ha='right')
    axes[1,0].axhline(0, color='r', linestyle='--')
    axes[1,0].axhline(30, color='orange', linestyle=':', alpha=0.5)
    axes[1,0].axhline(-30, color='orange', linestyle=':', alpha=0.5)
    axes[1,0].set_ylabel('Phase Difference from Position Error (deg)')
    axes[1,0].set_title('Phase Consistency Check')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # Summary table as text
    axes[1,1].axis('off')
    table_text = "Summary:\n" + "-"*50 + "\n"
    table_text += f"{'Pulsar':<12} {'A (μs)':<10} {'β (°)':<8} {'CMB (°)':<8} {'Δφ (°)':<8}\n"
    table_text += "-"*50 + "\n"
    for r in all_results:
        table_text += f"{r['pulsar_name']:<12} {r['A1_us']:<10.1f} "
        table_text += f"{r['coordinates']['ecl_lat_deg']:<8.1f} "
        table_text += f"{r['coordinates']['cmb_angle_deg']:<8.1f} "
        table_text += f"{r['phase_diff_deg']:<8.1f}\n"
    
    axes[1,1].text(0.1, 0.9, table_text, transform=axes[1,1].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_comparison.png', dpi=150)
    
    return fig

def full_analysis_pipeline(pulsar_configs, output_dir='.'):
    """Run complete analysis pipeline on multiple pulsars."""
    all_results = []
    
    for config in pulsar_configs:
        print(f"\n{'='*70}")
        print(f"Analyzing {config['name']}...")
        print('='*70)
        
        try:
            results = analyze_single_pulsar(config['par'], config['tim'], config['name'])
            
            if results is None:
                continue
            
            print_analysis_summary(results)
            
            # Individual pulsar plots
            prefix = f"{output_dir}/{config['name']}"
            plot_single_pulsar_analysis(results, save_prefix=prefix)
            
            # Frequency analysis
            freq_results = analyze_frequency_dependence(results)
            if freq_results:
                print("\nFrequency Analysis:")
                for fr in freq_results:
                    print(f"  {fr['freq_band']}: {fr['A1_us']:.1f} ± {fr['A1_err_us']:.1f} μs")
                plot_frequency_analysis(results, freq_results, save_prefix=prefix)
            
            # Solar elongation
            plot_solar_elongation(results, save_prefix=prefix)
            
            all_results.append(results)
            
        except Exception as e:
            print(f"Error analyzing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Multi-pulsar comparison
    if len(all_results) >= 2:
        print("\n" + "="*70)
        print("MULTI-PULSAR COMPARISON")
        print("="*70)
        compare_multiple_pulsars(all_results, save_prefix=f"{output_dir}/multi_pulsar")
    
    plt.show()
    
    return all_results


if __name__ == "__main__":
    # Example configuration - modify paths as needed
    pulsar_configs = [
        # {
            # 'name': 'J1713+0747',
            # 'par': 'NANOGrav15yr_PulsarTiming_v2.1.0/wideband/par/J1713+0747_PINT_20230131.wb.par',
            # 'tim': 'NANOGrav15yr_PulsarTiming_v2.1.0/wideband/tim/J1713+0747_PINT_20230131.wb.tim'
            # 'par': 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/par/J1713+0747_PINT_20220309.nb.par',
            # 'tim': 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/tim/J1713+0747_PINT_20220309.nb.tim'
        # },
        # {
            # 'name': 'J1744-1134',
            # 'par': 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/par/J1744-1134_PINT_20220302.nb.par',
            # 'tim': 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/tim/J1744-1134_PINT_20220302.nb.tim'
        # },
        {
            'name': 'J0030+0451',
            'par': 'NANOGrav15yr_PulsarTiming_v2.1.0/wideband/par/J0030+0451_PINT_20230131.wb.par',
            'tim': 'NANOGrav15yr_PulsarTiming_v2.1.0/wideband/tim/J0030+0451_PINT_20230131.wb.tim'
            # 'par': 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/par/J0030+0451_PINT_20220302.nb.par',
            # 'tim': 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/tim/J0030+0451_PINT_20220302.nb.tim'
        },
        # {
            # 'name': 'J0613-0200',
            # 'par': 'NANOGrav15yr_PulsarTiming_v2.1.0/wideband/par/J0613-0200_PINT_20230131.wb.par',
            # 'tim': 'NANOGrav15yr_PulsarTiming_v2.1.0/wideband/tim/J0613-0200_PINT_20230131.wb.tim',
            # 'par': 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/par/J0613-0200_PINT_20220302.nb.par'
            # 'tim': 'NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/tim/J0613-0200_PINT_20220302.nb.tim'
        # },
        # Add more pulsars here
    ]
    
    results = full_analysis_pipeline(pulsar_configs, output_dir='.')
