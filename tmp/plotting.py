import matplotlib.pyplot as pl
import pymc3 as pm
import corner
import utils
import numpy as np
from pandas.plotting import table
import os
import itertools
import datetime

def cornerplot(lc, trace, catalog, **kwargs):
    truths = pm.summary(trace)['mean']
    samples = pm.trace_to_dataframe(trace)
    cornerplot = corner.corner(samples, truths=truths, **kwargs);
    pl.annotate("{0} {1}".format(catalog, lc.id), xy=(0.4, 0.95), 
                     xycoords="figure fraction", fontsize=30)
    return cornerplot

def lcplot(ax, 
           lc, 
           trace=None,
           detrend_order=2,
           plot_outliers=False, 
           plot_trend=False, 
           uncertainties=False,
           highlight_color='r', 
           vline_color='b', 
           sigmaclip_kernel_size=5,
           sigmaclip_sigma=5,
           **kwargs):
    ax.plot(lc.t, lc.flux, '.', color='k')
    trend = utils.trend(lc.t, lc.flux, detrend_order)
    mask = utils.clipmask(lc.t, lc.flux, kernel_size=sigmaclip_kernel_size, sigma=sigmaclip_sigma)
    if plot_outliers:
        ax.plot(lc.t[mask], lc.flux[mask], '.', color=highlight_color, label='masked outliers')
    if plot_trend:
        ax.plot(lc.t, trend, color=highlight_color, label='order {0} polynomial fit'.format(detrend_order))
    if trace is not None:
        period = utils.modes(trace)['P']
        low, high = pm.stats.hpd(trace['P'], credible_interval=0.67)
        for i in np.arange(np.int((lc.t[-1]-lc.t[0]) / period)+2):
            if not uncertainties:
                ax.axvline(min(lc.t) + i*period, 
                       alpha=0.3, linewidth=2, color=vline_color)
            if uncertainties:
                vlines_min = min(lc.t)+i*low
                vlines_max = min(lc.t)+i*high
                ax.axvspan(vlines_min, vlines_max, alpha=0.3, color=vline_color)
    ax.set_xlim((min(lc.t), max(lc.t)))
    return ax
    
def plotsummary(ax, lc, trace, **kwargs):
    summary = pm.summary(trace)
    summary['mode'] = list(utils.modes(trace).values())
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    tbl = table(ax, summary.round(3), **kwargs)
    tbl.set_fontsize(30)
    return tbl

def writesummary(lc, trace, campaign, file):
    summary = pm.summary(trace)
    summary['mode'] = list(utils.modes(trace).values())
    columns = [c + "_" + n for c, n in 
               itertools.product(list(summary.index), 
                                 list(summary.columns))]
    columnstring = '\t'.join(columns)
    columnstring += '\tepic'
    columnstring += '\tcampaign\n'
    if not os.path.isfile(file):
            with open(file, "w") as f:
                f.write("Generated with round.py version 0.1 on " +
                        "{0}\n".format(datetime.datetime.now()
                                       .strftime("%Y-%m-%d %H:%M")))
                f.write("Campaign: {0}\n".format(campaign))
                f.write(columnstring)
    summarystring = summary.to_numpy().flatten()
    summarystring = np.append(summarystring, lc.id)
    summarystring = np.append(summarystring, campaign)
    summarystring = summarystring[None, :]
    fmtstring = ["%0.3f"]*(summarystring.size-2) + ["%d"]*2
    with open(file, "a+") as f: 
        np.savetxt(f, summarystring, fmt=fmtstring, delimiter='\t')
    return summarystring
   
def lcplot_withsummary(axs, lc, trace, 
                       summary_kwargs={}, lc_kwargs={}):
    lcplot(axs[0], lc, trace=trace)
    axs[1].xaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)
    table(axs[1], pm.summary(trace).round(3))
    pl.tight_layout()
    return axs

def acfplot_withsummary(axs, lc, trace, 
                        summary_kwargs={}, acf_kwargs={}):
    summary = pm.summary(trace)
    summary['mode'] = list(utils.modes(trace).values())
    plotacf(axs[0], lc, **acf_kwargs)
    axs[1].xaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)
    table(axs[1], summary.round(3))
    pl.tight_layout()
    return axs

def summaryplot(lc, 
                trace, 
                catalog,
                plot_prior=False,
                plot_period=False,
                summary_kwargs={}, 
                lc_kwargs={}, 
                acf_kwargs={},
                prior_kwargs={},
                outfile=None):
    fig, axs = pl.subplots(3, 1, figsize=(8.5, 11), 
                           gridspec_kw={'height_ratios': [1, 1, 0.001]})
    lcplot(axs[0], lc, trace=trace, detrend_order=2, **lc_kwargs)
    acfplot_withsummary(axs[1:], lc, trace,
                        summary_kwargs=summary_kwargs, 
                        acf_kwargs=acf_kwargs)
    if plot_period:
        axs[1].axvline(utils.varmode(trace, 'P'), color='k', linewidth=3, 
                       linestyle='--', label='MCMC period')
    if plot_prior:
        plotprior(axs[1], lc, label='prior', **prior_kwargs)
    axs[0].set_ylabel('everest flux', fontsize=15)
    axs[0].set_xlabel('time (BJD - 2454833)', fontsize=15)
    axs[0].legend()
    axs[1].set_ylabel('ACF', fontsize=15)
    axs[1].set_xlabel('lag (days)', fontsize=15)
    axs[1].legend()
    axs[0].set_title("\n{0} {1}".format(catalog, lc.id), fontsize=30)
    pl.tight_layout()
    if outfile is not None:
        pl.savefig(outfile)
    return fig
    
def plotacf(ax, lc, peaks=True, detrend_order=2, acf_kwargs={}, pk_kwargs={}):
    lag, power = utils.autocorr(lc.t, lc.flux - utils.trend(lc.t, lc.flux, detrend_order))
    pks, wts, _ = utils.candidates(lag, power)
    ax.plot(lag, power, label='ACF', **acf_kwargs)
    if peaks and (len(pks) > 0):
        if 'alpha' in pk_kwargs:
            wts = pk_kwargs['alpha'] * np.ones_like(w)
        else:
            wts /= max(wts)
        for i, (p, w) in enumerate(zip(pks, wts)):
            if i == 0:
                ax.axvline(p, alpha=w, **pk_kwargs, label='candidate periods')
            ax.axvline(p, alpha=w, **pk_kwargs)
    return ax

def plotprior(ax, lc, normalize=False, **kwargs):
    P = utils.periodprior(lc.t, lc.flux)
    x = np.linspace(0, max(lc.t) - min(lc.t), 1000)
    y = np.exp(P.logp(x).eval())
    if normalize:
        y = y / np.max(y)
    return ax.plot(x, y, **kwargs)
