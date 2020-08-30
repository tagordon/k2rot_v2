import pymc3 as pm
import distributions
import theano.tensor as tt
import exoplanet as xo
from lightcurve import Lightcurve
import utils
from scipy.signal import medfilt
from astropy.stats import sigma_clip
import plotting
import numpy as np
import sys
import os
import matplotlib.pyplot as pl

red = '#FE4365'
blue = '#00A9FF'
yellow = '#ECA25C'
green = '#3F9778'
darkblue = '#005D7F'

indir = sys.argv[1]
outdir = sys.argv[2]
outname = sys.argv[3]
ckptid = outname.split("_")[1]
ckptfile = ".ckpt_tess_{0}".format(ckptid)
ckpt = np.genfromtxt(indir + "/" + ckptfile, dtype=str)
files = ckpt[:, 0][ckpt[:, 1] == '0']
fileids = np.array(range(len(ckpt)))[ckpt[:, 1] == '0']

nf = len(ckpt)
outfile = outdir + "/" + outname
for i, f in zip(fileids, files):
    ticid = f.split("-")[2][-9:]
    camp = f.split("-")[1][-2:]
    print("computing {0} of {1} lightcurves\n TIC {2}".format(i+1, nf, ticid))
    summaryfile = ticid + "_" + camp + "_summary.png"
    cornerfile = ticid + "_" + camp + "_corner.png"
    errfile = ticid + "_" + camp + "_err.dat"
    lc = Lightcurve.tess(f)
    try:
        y = lc.flux - utils.trend(lc.t, lc.flux, 2)
        mask = sigma_clip(y-medfilt(y, kernel_size=301), sigma=3)
        x = lc.t[mask.mask == False]
        y = y[mask.mask == False]
        yerr = np.std(y - medfilt(y, kernel_size=51))
        x = x[::5]
        y = y[::5]
        
        with pm.Model() as model:
    
            mean = pm.Normal("mean", mu=np.mean(y), sd=np.std(y))
            yerr = pm.Normal("yerr", mu=yerr, sd=5.0)
            logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5)
            period = utils.periodprior(lc.t, lc.flux)
            logQ0 = pm.Uniform("logQ0", lower=-10, upper=10)
            logdQ = pm.Normal("logdQ", mu=2.0, sd=5.0)
            mix = pm.Uniform("mix", lower=0.0, upper=1.0)
    
            logS0 = pm.Normal("logS0", mu=np.log(np.var(y)), sd=5)
            dt = lc.t[-1] - lc.t[0]
            logw = pm.Uniform("logw", lower=-20, upper=0)
    
            kernel = xo.gp.terms.RotationTerm(
                log_amp=logamp,
                period=period,
                log_Q0=logQ0,
                log_deltaQ=logdQ,
                mix=mix
            )
            kernel += xo.gp.terms.SHOTerm(
                log_S0 = logS0,
                log_w0 = logw,
                log_Q = -np.log(np.sqrt(2))
            )                  
                      
            gp = xo.gp.GP(kernel, x, yerr**2 * np.ones_like(x), mean=mean, J=6)
            gp.marginal("gp", observed = y)

            start = model.test_point
            map_soln = xo.optimize(start=start, verbose=True)
            trace = pm.sample(
                tune=1000,
                draws=1000,
                start=start,
                cores=28,
                chains=28,
                step=xo.get_dense_nuts_step(target_accept=0.9)
            )
            
        plotting.cornerplot(lc, trace, 'TIC', smooth=True, truth_color=red);
        pl.savefig("{0}/{1}".format(outdir, cornerfile), dpi=200)
        
        acf_kwargs = {'color': 'k', 'linewidth': 3}
        pk_kwargs = {'color': red, 'linewidth': 1, 'linestyle': '--'}
        all_acf_kwargs = {'peaks': True, 'acf_kwargs': acf_kwargs, 'pk_kwargs': pk_kwargs}

        lc_kwargs = {'plot_outliers': True, 'plot_trend': True, 'uncertainties': False, 
             'highlight_color': blue, 'vline_color': blue, 
             'sigmaclip_kernel_size': 301, 'sigmaclip_sigma': 3}
        prior_kwargs = {'color': red, 'linewidth': 2, 'normalize': True}
        prior_kwargs = {'color': red, 'linewidth': 2, 'normalize': True}
        plotting.summaryplot(lc, trace, 'TIC', plot_period=True, plot_prior=True,
                             acf_kwargs=all_acf_kwargs, 
                             lc_kwargs=lc_kwargs, 
                             prior_kwargs=prior_kwargs);
        pl.savefig("{0}/{1}".format(outdir, summaryfile), dpi=200)
        plotting.writesummary(lc, trace, camp, outfile)
    except Exception as e:
        with open(errfile, "w") as errf:
            errf.write(repr(e))
    ckpt[i, 1] = '1'
    np.savetxt(indir + "/" + ckptfile, ckpt, fmt="%s")
