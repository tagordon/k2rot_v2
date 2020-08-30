import numpy as np
from astropy.io import fits 
import exoplanet as xo
import everest

class Lightcurve():
    
    def __init__(self, t, flux, identifier, fluxerr=None):
        self.flux = flux
        self.t = t
        self.id = identifier
        self.fluxerr = fluxerr
    
    @classmethod
    def everest(cls, file, corrected=True):
        with fits.open(file) as hdus:
            data = hdus[1].data
            hdr = hdus[1].header
        t = data['TIME']
        if corrected:
            flux = data['FCOR']
        else:
            flux = data['FLUX']
        m = ((data['QUALITY'] == 0) & 
             np.isfinite(t) & 
             np.isfinite(flux))
        t = np.ascontiguousarray(t[m], dtype=np.float64)
        flux = np.ascontiguousarray(flux[m], dtype=np.float64)
        identifier = hdr['KEPLERID']
        return cls(t, flux, identifier)
    
    @classmethod
    def epic(cls, epicid, **kwargs):
        try:
            file = everest.user.DownloadFile(epicid, **kwargs)
            return cls.everest(file)
        except AttributeError as e:
            print(e)
            
    @classmethod
    def tess(cls, file, **kwargs):
        with fits.open(file) as hdus:
            data = hdus[1].data
            hdr = hdus[1].header
        t = data['TIME']
        flux = data['PDCSAP_FLUX']
        m = ((data['QUALITY'] == 0) & 
             np.isfinite(t) & 
             np.isfinite(flux))
        t = np.ascontiguousarray(t[m], dtype=np.float64)
        flux = np.ascontiguousarray(flux[m], dtype=np.float64)
        identifier = hdr['TICID']
        return cls(t, flux, identifier)