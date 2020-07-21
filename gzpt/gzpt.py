#tie stuff together around the hzpt object
import numpy as np
import warnings
from . import zel
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit import cosmology
import os.path as path

class hzpt(object):
    """
    Parent object for the extended HZPT model for Power spectrum and 2PCF for dark matter and tracers.
    Each hzpt instance corresponds to one cosmology and one (effective) redshift.

    Parameters
    ----------
    cosmo: (cosmology object)
    z: redshift, float

    Methods
    ----------
    __call__


    Examples
    --------
    >>> k = np.logspace(-3, 3, num=60, endpoint=False)
    >>> c = nbodykit.cosmology.Planck15
    >>> z = 0.5
    >>> model = gzpt.hzpt(cosmo=c,z=z)
    >>> Pmm = matter.Correlator(model,nmax=1).Power()(k)
    >>> Pgg = tracers.AutoCorrelator(model,nmax=1).Power()(k)
    >>> Pgm = tracers.CrossCorrelator(model,nmax=2).Power()(k)
    ^Above is very ugly, will change this.

    >>>kdata,Pdata = np.loadtxt('some_Pk.dat',unpack=True)
    >>>Pmm.Pfit(kdata,Pdata)
    >>>Pmm_new = Pmm.Pfit(kdata,Pdata)

    """
    #FOR NOW will only support single z, but can come back to this later
    def __init__(self,cosmo,z):
        self.cosmo = cosmo
        self.z = z

