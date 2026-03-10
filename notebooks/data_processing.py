from lsst.rsp import get_tap_service
from lsst.rsp.service import get_siav2_service
from lsst.rsp.utils import get_pyvo_auth

import lsst.afw.display as afwDisplay
from lsst.afw.image import ExposureF
from lsst.afw.math import Warper, WarperConfig
from lsst.afw.fits import MemFileManager
import lsst.geom as geom

from pyvo.dal.adhoc import DatalinkResults, SodaQuery

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from astropy import units as u
from astropy.table import Table, vstack
import corner
from glob import glob
from astropy.coordinates import SkyCoord
import os

import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

BAND_COLORS = {
    'u': '#0c71ff',
    'g': '#49be61',
    'r': '#c61c00',
    'i': '#ffc200',
    'z': '#f341a2',
    'y': '#5d0000'
}

BAND_ORDER = ['u', 'g', 'r', 'i', 'z', 'y']

FIELD_TITLES = {
    'ecdfs': 'ECDFS',
    'galactic': 'Low Galactic Latitude Field',
    'ecliptic': 'Low Ecliptic Latitude Field',
}

def get_title(table):
    """Return a display title based on which fields are present in the table."""
    if 'field' not in table.colnames:
        return 'Unknown Field'
    fields = set(table['field'])
    if len(fields) > 1:
        return 'All 3 Fields'
    return FIELD_TITLES.get(fields.pop(), 'Unknown Field')

service = get_tap_service("tap")
assert service is not None

sia_service = get_siav2_service("dp1")
assert sia_service is not None

def load_dp1(
    fetch_from_server: bool = False,
    load_all_fields: bool = True,
    field: str = 'ecdfs',
    service = None,
    data_dir: str = 'dp1_fields'
) -> Table:
    """
    Load astronomical data and return results table.

    Parameters
    ----------
    fetch_from_server : bool
        True: Query TAP service | False: Load from .fits files
    load_all_fields : bool
        True: Merge all fields | False: Single field only
    field : str
        Field to load if load_all_fields=False ('ecdfs', 'galactic', 'ecliptic')
    service : optional
        TAP service object (required if fetch_from_server=True)
    data_dir : str
        Directory to save/load FITS files (default: 'dp1_fields')

    Returns
    -------
    Table
        Astropy Table with results
    """
    ALL_FIELDS = ['ecdfs', 'galactic', 'ecliptic']

    FIELD_COORDS = {
        'ecdfs': (53.16, -28.10),
        'galactic': (95.0, -25.0),
        'ecliptic': (37.98, 7.015)
    }

    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    def fetch_field_data(field_name):
        """Fetch data from server for a given field."""
        ra_cen, dec_cen = FIELD_COORDS[field_name]

        query = """SELECT apFlux, apFlux_flag, apFlux_flag_apertureTruncated, apFluxErr,
            band, bboxSize, centroid_flag, coord_dec, coord_ra,
            dec, decErr, detector, diaObjectId, diaSourceId,
            dipoleAngle, dipoleChi2, dipoleFitAttempted, dipoleFluxDiff, dipoleFluxDiffErr,
            dipoleLength, dipoleMeanFlux, dipoleMeanFluxErr, dipoleNdata,
            extendedness, forced_PsfFlux_flag, forced_PsfFlux_flag_edge,
            forced_PsfFlux_flag_noGoodPixels, isDipole,
            ixx, ixxPSF, ixy, ixyPSF, iyy, iyyPSF,
            midpointMjdTai, parentDiaSourceId,
            pixelFlags, pixelFlags_bad, pixelFlags_cr, pixelFlags_crCenter,
            pixelFlags_edge, pixelFlags_injected, pixelFlags_injected_template,
            pixelFlags_injected_templateCenter, pixelFlags_injectedCenter,
            pixelFlags_interpolated, pixelFlags_interpolatedCenter,
            pixelFlags_nodata, pixelFlags_nodataCenter, pixelFlags_offimage,
            pixelFlags_saturated, pixelFlags_saturatedCenter,
            pixelFlags_streak, pixelFlags_streakCenter,
            pixelFlags_suspect, pixelFlags_suspectCenter,
            psfChi2, psfFlux, psfFlux_flag, psfFlux_flag_edge,
            psfFlux_flag_noGoodPixels, psfFluxErr, psfNdata,
            ra, ra_dec_Cov, raErr, reliability,
            scienceFlux, scienceFluxErr,
            shape_flag, shape_flag_no_pixels, shape_flag_not_contained,
            shape_flag_parent_source, snr, ssObjectId,
            trail_flag_edge, trailAngle, trailDec, trailFlux, trailLength, trailRa,
            visit, x, xErr, y, yErr
            FROM dp1.DiaSource
            WHERE CONTAINS(POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {}, {}, 1.0)) = 1
            ORDER BY diaSourceId ASC""".format(ra_cen, dec_cen)

        print(f"Querying {field_name}...")
        print(query)

        job = service.submit_job(query)
        job.run()
        job.wait(phases=['COMPLETED', 'ERROR'])
        print(f'Job phase is {job.phase}')

        if job.phase == 'ERROR':
            job.raise_if_error()
        assert job.phase == 'COMPLETED'

        data = job.fetch_result().to_table()
        print(f"Retrieved {len(data)} rows with {len(data.colnames)} columns")

        # Save to dp1_fields directory
        filepath = os.path.join(data_dir, f'{field_name}.fits')
        data.write(filepath, format='fits', overwrite=True)
        print(f"Saved {len(data)} rows to {filepath}")

        return data

    # Main logic
    if fetch_from_server:
        if service is None:
            raise ValueError("service parameter is required when fetch_from_server=True")

        if load_all_fields:
            all_results = []
            for f in ALL_FIELDS:
                result = fetch_field_data(f)
                result['field'] = f
                all_results.append(result)
            results = vstack(all_results)
            print(f"Merged {len(results)} total rows from all fields")
        else:
            results = fetch_field_data(field)
            results['field'] = field
    else:
        if load_all_fields:
            all_results = []
            for f in ALL_FIELDS:
                filepath = os.path.join(data_dir, f'{f}.fits')
                data = Table.read(filepath)
                print(f"Loaded {len(data)} rows from {filepath}")
                data['field'] = f
                all_results.append(data)
            results = vstack(all_results)
            print(f"Merged {len(results)} total rows from all fields")
        else:
            filepath = os.path.join(data_dir, f'{field}.fits')
            results = Table.read(filepath)
            results['field'] = field
            print(f"Loaded {len(results)} rows from {filepath}")

    return results

def add_engineered_features(table):
    """
    Add engineered features to the table.
    
    Parameters
    ----------
    table : Table
        Astropy Table to add features to
    
    Returns
    -------
    Table
        Table with added engineered features
    """
    # Reorder columns to put diaSourceId first
    names = table.colnames
    names.remove('diaSourceId')
    names.insert(0, 'diaSourceId')
    table = table[names]
    
    # Engineered features
    table['flux_ext'] = table['apFlux'] / table['psfFlux']
    
    table['ellip_ext'] = (
        (np.sqrt((table['ixx'] - table['iyy'])**2 + 4 * table['ixy']**2) / 
         (table['ixx'] + table['iyy'])) - 
        (np.sqrt((table['ixxPSF'] - table['iyyPSF'])**2 + 4 * table['ixyPSF']**2) / 
         (table['ixxPSF'] + table['iyyPSF']))
    )
    
    table['i_ext'] = (table['ixx'] + table['iyy']) / (table['ixxPSF'] + table['iyyPSF'])
    
    table['template_flux'] = table['scienceFlux'] - table['psfFlux']
    table['temp_sci_flux_ratio'] = table['template_flux'] / table['scienceFlux']
    
    # For FWHM circle on plot (converted to pixels)
    table['psf_fwhm'] = (table['ixxPSF'] * table['iyyPSF'] - table['ixyPSF']**2)**(1/4) * 2.35482 * 5
    
    return table

from matplotlib.patches import Circle
from astropy.visualization import ZScaleInterval
import requests
from io import BytesIO
from PIL import Image


def get_cutout_with_retry(dl_result, spherePoint, session, fov, max_retries=3):
    """Get a cutout with exponential backoff retry logic."""
    sq = SodaQuery.from_resource(dl_result,
                                 dl_result.get_adhocservice_by_id("cutout-sync-exposure"),
                                 session=session)
    sphereRadius = fov * u.deg
    sq.circle = (spherePoint.getRa().asDegrees() * u.deg,
                 spherePoint.getDec().asDegrees() * u.deg,
                 sphereRadius)
    
    for attempt in range(max_retries):
        try:
            cutout_bytes = sq.execute_stream().read()
            sq.raise_if_error()
            mem = MemFileManager(len(cutout_bytes))
            mem.setData(cutout_bytes, len(cutout_bytes))
            return ExposureF(mem)
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise


def fetch_images_for_row(row, sia_service, get_pyvo_auth, fov=0.003):
    """Fetch all images for a single row with rate limiting."""
    try:
        row_start = time.time()
        
        ra = row['ra']
        dec = row['dec']
        visit = row['visit']
        band = row['band']
        diaSourceId = row['diaSourceId']
        snr = row['snr']
        extendedness = row['extendedness']
        flux_ext = row['flux_ext']
        ellip_ext = row['ellip_ext']
        i_ext = row['i_ext']
        template_flux = row['template_flux']
        scienceFlux = row['scienceFlux']
        psfFlux = row['psfFlux']
        apFlux = row['apFlux']
        psf_fwhm = row['psf_fwhm']
        
        spherePoint = geom.SpherePoint(ra*geom.degrees, dec*geom.degrees)
        circle = (ra, dec, 0.0001)
        
        t0 = time.time()
        lvl2_table = sia_service.search(pos=circle, calib_level=2).to_table()
        sel = lvl2_table['dataproduct_subtype'] == 'lsst.visit_image'
        sel &= lvl2_table['lsst_visit'] == visit
        sci_table = lvl2_table[sel]
        
        if len(sci_table) == 0:
            return None, f"No science images found"
        
        lvl3_table = sia_service.search(pos=circle, calib_level=3).to_table()
        search_time = time.time() - t0
        
        sel = lvl3_table['dataproduct_subtype'] == 'lsst.template_coadd'
        sel &= lvl3_table['lsst_band'] == band
        ref_table = lvl3_table[sel]
        
        if len(ref_table) == 0:
            return None, f"No template images found"
        
        sel = lvl3_table['dataproduct_subtype'] == 'lsst.difference_image'
        sel &= lvl3_table['lsst_visit'] == visit
        diff_table = lvl3_table[sel]
        
        if len(diff_table) == 0:
            return None, f"No difference images found"
        
        t0 = time.time()
        dl_result_sci = DatalinkResults.from_result_url(sci_table['access_url'][0], session=get_pyvo_auth())
        dl_result_ref = DatalinkResults.from_result_url(ref_table['access_url'][0], session=get_pyvo_auth())
        dl_result_diff = DatalinkResults.from_result_url(diff_table['access_url'][0], session=get_pyvo_auth())
        datalink_time = time.time() - t0
        
        t0 = time.time()
        sci = get_cutout_with_retry(dl_result_sci, spherePoint, get_pyvo_auth(), fov)
        time.sleep(0.3)
        ref = get_cutout_with_retry(dl_result_ref, spherePoint, get_pyvo_auth(), fov)
        time.sleep(0.3)
        diff = get_cutout_with_retry(dl_result_diff, spherePoint, get_pyvo_auth(), fov)
        download_time = time.time() - t0
        
        t0 = time.time()
        warper_config = WarperConfig()
        warper = Warper.fromConfig(warper_config)
        sci_wcs = sci.getWcs()
        sci_bbox = sci.getBBox()
        warped_ref = warper.warpExposure(sci_wcs, ref, destBBox=sci_bbox)
        warp_time = time.time() - t0
        
        total_time = time.time() - row_start
        
        return {
            'visit': visit,
            'band': band,
            'diaSourceId': diaSourceId,
            'ra': ra,
            'dec': dec,
            'sci': sci,
            'snr': snr,
            'extendedness': extendedness,
            'flux_ext': flux_ext,
            'ellip_ext': ellip_ext,
            'i_ext': i_ext,
            'template_flux': template_flux,
            'scienceFlux': scienceFlux,
            'psfFlux': psfFlux,
            'apFlux': apFlux,
            'psf_fwhm': psf_fwhm,
            'warped_ref': warped_ref,
            'diff': diff,
            'search_time': search_time,
            'datalink_time': datalink_time,
            'download_time': download_time,
            'warp_time': warp_time,
            'total_time': total_time
        }, None
        
    except Exception as e:
        return None, str(e)


def fetch_legacy_survey_cutout(ra, dec, pixscale=0.15, timeout=10, max_retries=3):
    """Fetch a cutout from Legacy Survey with retry logic."""
    url = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&layer=ls-dr9&pixscale={pixscale}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return np.array(img), None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 1.0 + attempt
                print(f"    ⚠ Retry {attempt + 1}/{max_retries - 1} after {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                return None, str(e)
    
    return None, "Max retries exceeded"


def create_image_gallery(table, rows=5, cols=2, include_legacy=True):
    """
    Create a gallery of LSST difference imaging triplets and optionally Legacy Survey cutouts.
    
    Parameters
    ----------
    table : Table
        Astropy Table containing source data with required columns:
        ra, dec, visit, band, diaSourceId, snr, extendedness, flux_ext, 
        ellip_ext, i_ext, template_flux, scienceFlux, psfFlux, apFlux, psf_fwhm
    rows : int
        Number of rows in the gallery layout (default: 5)
    cols : int
        Number of columns in the gallery layout (default: 2)
    include_legacy : bool
        Whether to create Legacy Survey cutout gallery (default: True)
    
    Returns
    -------
    tuple
        (fig_lsst, fig_legacy) - matplotlib Figure objects for LSST and Legacy Survey galleries
        If include_legacy=False, returns (fig_lsst, None)
    """
    
    n_images = cols * rows
    sample_indices = np.random.choice(len(table), size=min(n_images, len(table)), replace=False)
    sampled_rows = [table[i] for i in sample_indices]
    
    print(f"\n{'='*60}")
    print(f"Starting gallery creation at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Layout: {cols} columns x {rows} rows = {n_images} images")
    print(f"{'='*60}\n")
    start_time = time.time()
    
    # Fetch LSST images
    gallery_results = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_images_for_row, row, sia_service, get_pyvo_auth): idx 
                   for idx, row in enumerate(sampled_rows)}
        
        for future in as_completed(futures):
            idx = futures[future]
            result, error = future.result()
            
            if result:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ {len(gallery_results)+1}/{n_images}: visit={result['visit']}, diaSourceId={result['diaSourceId']}, time={result['total_time']:.1f}s")
                gallery_results.append(result)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed: {error}")
            
            if len(gallery_results) >= n_images:
                break
    
    fetch_time = time.time() - start_time
    print(f"\nAll images fetched in {fetch_time:.1f}s ({fetch_time/60:.1f} min)")
    print(f"Average per image set: {fetch_time/len(gallery_results):.1f}s\n")
    
    actual_n_images = len(gallery_results)
    if actual_n_images < n_images:
        print(f"\n⚠️  Warning: Only {actual_n_images} images available, but layout is {cols}x{rows}={n_images}")
        print(f"Remaining {n_images - actual_n_images} cells will be left empty/white\n")
    
    # Create LSST gallery plot
    print("Creating LSST plots...")
    plot_start = time.time()
    
    fig_lsst = plt.figure(figsize=(cols * 9, rows * 3))
    gs = GridSpec(rows, cols, figure=fig_lsst, 
                  hspace=0.02, wspace=0.02,
                  left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    for idx in range(n_images):
        gallery_row = idx // cols
        gallery_col = idx % cols
        
        gs_sub = gs[gallery_row, gallery_col].subgridspec(1, 3, wspace=0.01)
        
        ax1 = fig_lsst.add_subplot(gs_sub[0, 0])
        ax2 = fig_lsst.add_subplot(gs_sub[0, 1])
        ax3 = fig_lsst.add_subplot(gs_sub[0, 2])
        
        if idx < len(gallery_results):
            result = gallery_results[idx]
            
            for ax, img in [(ax1, result['sci']), (ax2, result['warped_ref']), (ax3, result['diff'])]:
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(img.image.array)
                
                ax.imshow(img.image.array, cmap='gray', origin='lower', 
                          vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')
                ax.set_axis_off()
                ax.set_position(ax.get_position())
                ax.margins(0, 0)
                ax.set_xlim(0, img.image.array.shape[1])
                ax.set_ylim(0, img.image.array.shape[0])
            
            ax1.text(0.02, 0.98, f'Visit: {result["visit"]}\nSNR: {result["snr"]:.3f}\nSci Flux: {result["scienceFlux"]:.1f}', 
                     transform=ax1.transAxes, ha='left', va='top',
                     fontsize=10, color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            ax1.text(0.02, 0.02, f'{result["band"]}', 
                     transform=ax1.transAxes, ha='left', va='bottom',
                     fontsize=16, color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
            ax2.text(0.02, 0.98, f'RA: {result["ra"]}\nDEC: {result["dec"]}', 
                     transform=ax2.transAxes, ha='left', va='top',
                     fontsize=10, color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
            ax2.text(0.02, 0.02, f'Template Flux: {result["template_flux"]:.1f} ({(result["template_flux"]/result["scienceFlux"]*100):.0f}% sci flux)', 
                     transform=ax2.transAxes, ha='left', va='bottom',
                     fontsize=10, color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            ax3.text(0.98, 0.98, f'DIASourceID: {result["diaSourceId"]}\nPSF Flux: {result["psfFlux"]:.1f} | Ap Flux: {result["apFlux"]:.1f}', 
                     transform=ax3.transAxes, ha='right', va='top',
                     fontsize=10, color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
            ax3.text(0.98, 0.02, f'Ext.: {result["extendedness"]:.3f}\nLog Flux Ext.: {np.log10(result["flux_ext"]):.3f}\nEllip. Diff.: {result["ellip_ext"]:.3f}\nMoment Ext.: {result["i_ext"]:.3f}', 
                     transform=ax3.transAxes, ha='right', va='bottom',
                     fontsize=10, color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
            psf_fwhm_pixels = result['psf_fwhm']
            circle_x = psf_fwhm_pixels * 1.5
            circle_y = psf_fwhm_pixels * 1.5
            
            circle = Circle((circle_x, circle_y), psf_fwhm_pixels / 2, 
                           fill=False, edgecolor='cyan', linewidth=2, alpha=0.8)
            ax3.add_patch(circle)
            
            cross_length = psf_fwhm_pixels / 2
            ax3.plot([circle_x - cross_length, circle_x + cross_length], 
                    [circle_y, circle_y], 
                    color='cyan', linewidth=2, alpha=0.8)
            ax3.plot([circle_x, circle_x], 
                    [circle_y - cross_length, circle_y + cross_length], 
                    color='cyan', linewidth=2, alpha=0.8)
            
        else:
            for ax in [ax1, ax2, ax3]:
                ax.set_axis_off()
    
    plot_time = time.time() - plot_start
    print(f"LSST plotting complete in {plot_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"LSST gallery complete at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Created {len(gallery_results)} image sets")
    print(f"{'='*60}\n")
    
    fig_legacy = None
    
    if include_legacy:
        # Create Legacy Survey gallery
        print("\n" + "="*60)
        print("Creating Legacy Survey cutout gallery...")
        print("="*60 + "\n")
        
        print("Downloading Legacy Survey cutouts...")
        legacy_start = time.time()
        legacy_images = []
        
        for idx, result in enumerate(gallery_results):
            img, error = fetch_legacy_survey_cutout(result['ra'], result['dec'])
            if img is not None:
                legacy_images.append(img)
                print(f"✓ {idx+1}/{len(gallery_results)}: Downloaded cutout for DIASourceID {result['diaSourceId']}")
            else:
                print(f"✗ {idx+1}/{len(gallery_results)}: Failed to download - {error}")
                legacy_images.append(None)
            time.sleep(0.1)
        
        legacy_fetch_time = time.time() - legacy_start
        print(f"\nLegacy Survey cutouts downloaded in {legacy_fetch_time:.1f}s\n")
        
        fig_legacy = plt.figure(figsize=(cols * 3, rows * 3))
        gs_legacy = GridSpec(rows, cols, figure=fig_legacy,
                             hspace=0.02, wspace=0.02,
                             left=0.01, right=0.99, top=0.99, bottom=0.01)
        
        for idx in range(n_images):
            gallery_row = idx // cols
            gallery_col = idx % cols
            
            ax = fig_legacy.add_subplot(gs_legacy[gallery_row, gallery_col])
            
            if idx < len(legacy_images) and legacy_images[idx] is not None:
                ax.imshow(legacy_images[idx], origin='upper', aspect='equal', interpolation='nearest')
                ax.set_axis_off()
                ax.margins(0, 0)
            else:
                ax.set_axis_off()
        
        legacy_plot_time = time.time() - legacy_start - legacy_fetch_time
        print(f"Legacy Survey plotting complete in {legacy_plot_time:.1f}s")
        
        total_legacy_time = time.time() - legacy_start
        print(f"Total Legacy Survey gallery time: {total_legacy_time:.1f}s\n")
        
        print("="*60)
        print("Both galleries complete")
        print("="*60)
    
    return fig_lsst, fig_legacy