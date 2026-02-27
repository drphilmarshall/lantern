"""LSST filter that uses alert DIASource information to find extended lensed AGN candidates. Designed to run on ANTARES."""

import math

BAND_FLUX_THRESHOLDS = {
    "u": 178286.7,
    "g": 255131.4,
    "r": 657111.2,
    "i": 743053.0,
    "z": 963585.1,
    "y": 1013290.7,
}


def _ellipticity(ixx, iyy, ixy):
    """e = sqrt((Ixx-Iyy)^2 + 4*Ixy^2) / (Ixx+Iyy). Ensures no division by zero."""
    denom = ixx + iyy
    if denom == 0.0:
        return 0.0
    return math.sqrt((ixx - iyy) ** 2 + 4.0 * ixy ** 2) / denom


def evaluate(props):
    """
    Run all cuts on alert properties dict.

    Returns dict of computed metrics if alert passes all cuts.
    Returns None if any cut fails.

    """
    psf_flux      = props["lsst_diaSource_psfFlux"]
    ap_flux       = props["lsst_diaSource_apFlux"]
    science_flux  = props["lsst_diaSource_scienceFlux"]
    template_flux = props["lsst_diaSource_templateFlux"]
    snr           = props["lsst_diaSource_snr"]
    band          = props["lsst_diaSource_band"]

    ixx     = props["lsst_diaSource_ixx"]
    iyy     = props["lsst_diaSource_iyy"]
    ixy     = props["lsst_diaSource_ixy"]
    ixx_psf = props["lsst_diaSource_ixxPSF"]
    iyy_psf = props["lsst_diaSource_iyyPSF"]
    ixy_psf = props["lsst_diaSource_ixyPSF"]

    # Quality cuts
    all_pos = psf_flux > 0.0 and ap_flux > 0.0 and science_flux > 0.0
    all_neg = psf_flux < 0.0 and ap_flux < 0.0 and science_flux < 0.0
    if any([
        props["lsst_diaSource_apFlux_flag"],
        props["lsst_diaSource_psfFlux_flag"],
        props["lsst_diaSource_pixelFlags_cr"],
        props["lsst_diaSource_pixelFlags_bad"],
        props["lsst_diaSource_pixelFlags_nodata"],
        props["lsst_diaSource_pixelFlags_interpolated"],
        props["lsst_diaSource_pixelFlags_saturated"],
        props["lsst_diaSource_pixelFlags_suspect"],
        snr <= 15.0,
        not (all_pos or all_neg),
    ]):
        return None

    # Extendedness cuts

    ## ensure no division by zero
    if psf_flux == 0.0 or science_flux == 0.0:
        return None
    psf_trace = ixx_psf + iyy_psf
    if psf_trace == 0.0:
        return None
    
    ## engineered extendedness features
    flux_ext          = ap_flux / psf_flux
    ellip_ext         = _ellipticity(ixx, iyy, ixy) - _ellipticity(ixx_psf, iyy_psf, ixy_psf)
    i_ext             = (ixx + iyy) / psf_trace
    temp_sci_flux_ratio = template_flux / science_flux

    ## cuts
    if flux_ext <= 1.259 or i_ext <= 1.5 or ellip_ext <= 0.2:
        return None

    # Moving object cut
    if temp_sci_flux_ratio <= 0.25:
        return None

    # Per-band template flux cut
    threshold = BAND_FLUX_THRESHOLDS.get(band)
    if threshold is None or template_flux >= threshold:
        return None

    return {
        "snr":               snr,
        "flux_ext":          flux_ext,
        "i_ext":             i_ext,
        "ellip_ext":         ellip_ext,
        "temp_sci_flux_ratio": temp_sci_flux_ratio,
    }
