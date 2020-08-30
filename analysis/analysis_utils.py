import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.interpolate import interp1d
import read_mist_models

def selectms(df, mistfile, dist_range=(0, 600), bp_rp_range=(0.5, 2.8), g_rp_range=(0.0, 3.0), width=(-0.2, 0.4)):
    
    iso = read_mist_models.ISOCMD(mistfile)
    mist = iso.isocmds[iso.age_index(9.0)]

    #good_parallax = df["parallax_error"] < 0.1
    #unimodal_distance_result = ((df["r_modality_flag"] == 1) 
    #                            & (df["r_result_flag"] == 1))
    has_finite_bp_rp = np.isfinite(df["bp_rp"])
    #good_bp = df["phot_bp_mean_flux_error"]/df[u'phot_bp_mean_flux'] < 0.1
    good_rp = df[u'phot_rp_mean_flux_error']/df[u'phot_rp_mean_flux'] < 0.01
    good_mg = df[u'phot_g_mean_flux_error']/df[u'phot_g_mean_flux'] < 0.01
    in_r_range = (df["r_est"] > dist_range[0]) & (df["r_est"] < dist_range[1])
    in_bp_rp_range = (df["bp_rp"] < bp_rp_range[1]) & df["bp_rp"] > bp_rp_range[0]
    in_g_rp_range = (df["g_rp"] < g_rp_range[1]) & (df["g_rp"] > g_rp_range[0])
    mask = (has_finite_bp_rp & good_rp 
            & good_mg & in_r_range & in_g_rp_range)

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