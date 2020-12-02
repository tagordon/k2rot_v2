import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.interpolate import interp1d
import read_mist_models
from scipy.ndimage import generate_binary_structure, binary_erosion, label
from scipy import ndimage as ndi

def selectms(df, mistfile, bp_rp_range=(0.5, 2.8), g_rp_range=(0.0, 3.0), width=(-0.2, 0.4)):
    
    iso = read_mist_models.ISOCMD(mistfile)
    mist = iso.isocmds[iso.age_index(9.0)]

    good_rp = df[u'phot_rp_mean_flux_error']/df[u'phot_rp_mean_flux'] < 0.01
    good_mg = df[u'phot_g_mean_flux_error']/df[u'phot_g_mean_flux'] < 0.01
    in_bp_rp_range = (df["bp_rp"] < bp_rp_range[1]) & df["bp_rp"] > bp_rp_range[0]
    in_g_rp_range = (df["g_rp"] < g_rp_range[1]) & (df["g_rp"] > g_rp_range[0])
    mask = (good_rp & good_mg & in_g_rp_range)

    iso_bp_rp = mist['Gaia_BP_MAWb'] - mist['Gaia_RP_MAW']
    iso_mg = mist['Gaia_G_MAW']
    mass_mask = (mist['initial_mass'] < 2.0) & (mist['initial_mass'] > 0.2)
    iso_bp_rp = iso_bp_rp[mass_mask]
    iso_mg = iso_mg[mass_mask]
    interpolator = interp1d(iso_bp_rp, iso_mg)
    in_color_range = (df["bp_rp"] > min(iso_bp_rp)) & (df["bp_rp"] < max(iso_bp_rp))
    mask = mask & in_color_range
    iso_mg_interp = interpolator(df[mask]['bp_rp'])
    correction = 5 * np.log10(df[mask]["r_est"]) - 5 
    bp_rp, mg = (np.array(df[mask]["bp_rp"]), 
                 np.array(df[mask]["phot_g_mean_mag"])
                 -correction)
    is_ms = (mg - iso_mg_interp < -width[0]) & (iso_mg_interp - mg < width[1])
    return df[is_ms & mask]

def detect_edges(z):

    gradx = z[0]
    grady = z[1]
    magnitude = np.linalg.norm(z, axis=0)
    mask = np.ones(magnitude.shape, dtype=bool)
    s = generate_binary_structure(2, 2)
    eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)

    local_maxima = np.zeros(magnitude.shape, bool)
    pts_plus = (gradx >= 0) & (grady >= 0) & (np.abs(gradx) >= np.abs(grady))
    pts_minus = (gradx <= 0) & (grady <= 0) & (np.abs(gradx) >= np.abs(grady))
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts

    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = np.abs(grady)[pts] / np.abs(gradx)[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    pts_plus = (gradx >= 0) & (grady >= 0) & (np.abs(gradx) <= np.abs(grady))
    pts_minus = (gradx <= 0) & (grady <= 0) & (np.abs(gradx) <= np.abs(grady))
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = np.abs(gradx)[pts] / np.abs(grady)[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    pts_plus = (gradx <= 0) & (grady >= 0) & (np.abs(gradx) <= np.abs(grady))
    pts_minus = (gradx >= 0) & (grady <= 0) & (np.abs(gradx) <= np.abs(grady))
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = np.abs(gradx)[pts] / np.abs(grady)[pts]
    c_plus = c2a * w + c1a * (1.0 - w) <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    pts_plus = (gradx <= 0) & (grady >= 0) & (np.abs(gradx) >= np.abs(grady))
    pts_minus = (gradx >= 0) & (grady <= 0) & (np.abs(gradx) >= np.abs(grady))
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = np.abs(grady)[pts] / np.abs(gradx)[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    high_threshold = 0.25
    low_threshold = 0.2
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)

    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                         np.arange(count, dtype=np.int32) + 1),
                 copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask

def g_rp_to_mass(g_rp, mistfile):
    
    mistfile = '../analysis/MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd'
    iso = read_mist_models.ISOCMD(mistfile)
    mist = iso.isocmds[iso.age_index(9.0)]
    
    iso_bp_rp = mist['Gaia_BP_MAWb'] - mist['Gaia_RP_MAW']
    iso_g_rp = mist['Gaia_G_MAW'] - mist['Gaia_RP_MAW']
    iso_mg = mist['Gaia_G_MAW']
    iso_mass = mist['initial_mass']
    mass = interp1d(iso_g_rp[iso_mass < 2], iso_mass[iso_mass < 2])
    g_rp_mask = (g_rp < np.max(iso_g_rp[iso_mass < 2])) & (g_rp > np.min(iso_g_rp[iso_mass < 2]))
    m = mass(g_rp[g_rp_mask])
    return m, g_rp_mask

def bp_rp_to_mass(bp_rp, mistfile):
    
    mistfile = '../analysis/MIST_v1.2_feh_p0.50_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd'
    iso = read_mist_models.ISOCMD(mistfile)
    mist = iso.isocmds[iso.age_index(9.0)]
    
    iso_bp_rp = mist['Gaia_BP_MAWb'] - mist['Gaia_RP_MAW']
    iso_g_rp = mist['Gaia_G_MAW'] - mist['Gaia_RP_MAW']
    iso_mg = mist['Gaia_G_MAW']
    iso_mass = mist['initial_mass']
    mass = interp1d(iso_bp_rp[iso_mass < 2], iso_mass[iso_mass < 2])
    bp_rp_mask = (bp_rp < np.max(iso_bp_rp[iso_mass < 2])) & (bp_rp > np.min(iso_bp_rp[iso_mass < 2]))
    m = mass(bp_rp[bp_rp_mask])
    return m, bp_rp_mask