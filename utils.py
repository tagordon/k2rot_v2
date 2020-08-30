import numpy as np
import exoplanet as xo
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter
from astropy.stats import sigma_clip
from scipy.signal import medfilt
import distributions
import pymc3 as pm

def trend(t, y, n=2):
    res = np.polyfit(t, y, n)
    trend = sum([c*(t**i) for (i, c) in enumerate(res[::-1])])
    return trend

def clipmask(t, y, n=2, kernel_size=5, sigma=5):
    y = y - trend(t, y)
    yfilt = y - medfilt(y, kernel_size=kernel_size)
    mask = sigma_clip(yfilt, sigma=sigma)
    return mask.mask

def autocorr(t, y, max_peaks=1, min_period=0.5):
    results = xo.autocorr_estimator(t, y, min_period=min_period)
    return results["autocorr"]

def peaks(lag, power, **kwargs):
    if not 'prominence' in kwargs:
        kwargs['prominence'] = 0.01
    if not 'width' in kwargs:
        kwargs['width'] = 1
    if not 'height' in kwargs:
        kwargs['height'] = 0.0
    cadence = lag[2]
    peaks, properties = find_peaks(power, **kwargs)
    perm = np.argsort(properties['peak_heights'])
    prominences = properties['prominences'][perm[::-1]]
    widths = properties['widths'][::-1]*cadence
    heights = properties['peak_heights'][perm[::-1]]
    peaks = lag[peaks[perm[::-1]]]
    return peaks, prominences, widths, heights

def candidates(lag, power):
    pks, prominences, widths, heights = peaks(lag, power)
    mask = [pks < max(lag)/2]
    pks = pks[mask]
    prominences = prominences[mask]
    widths = widths[mask]
    heights = heights[mask]
    if len(pks) > 10:
        pks = pks[:10]
        prominences = prominences[:10]
        widths = widths[:10]
        heights = heights[:10]
    sd = widths# / 2.35
    w = (prominences ** 0.5) * heights
    pks = np.concatenate((pks, 0.5*pks, 2*pks))
    w = np.concatenate((w, w, w))
    sd = np.concatenate((sd, sd, sd))
    w[pks > max(lag)/2] = 0
    return pks, w, sd

def periodprior(t, y, detrend_order=2, min_period=0.5):
    detrended_y = y - trend(t, y, detrend_order)
    lag, power = autocorr(t, detrended_y, min_period=min_period)
    pks, w, sd = candidates(lag, power)
    #w[pks > max(lag)/2] = 0
    w = w / np.sum(w)
    mu = pks
    if len(pks) == 0:
        try:
            dist = pm.Uniform("P", lower=0.0, upper=max(lag)/2)
        except TypeError:
            dist = pm.Uniform.dist(lower=0.0, upper=max(lag)/2)
        return dist
    try:
        dist = pm.NormalMixture("P", w=w, mu=mu, sd=sd, testval=mu[np.argmax(w)])
    except (TypeError, ValueError) as e:
        print(e)
        dist = pm.NormalMixture.dist(w=w, mu=mu, sd=sd)
    return dist

def varmode(trace, varname):
    bins, edges = np.histogram(trace[varname], bins=100)
    return edges[np.argmax(bins)]

def modes(trace):
    modedict = {}
    for v in trace.varnames:
        if not '_interval__' in v:
            modedict[v] = varmode(trace, v)
    return modedict

