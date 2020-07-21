#tie stuff together around the hzpt object
import numpy as np
import warnings
from zel import loginterp
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from nbodykit import cosmology
import os.path as path

class hzpt:
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
    >>>kdata,Pdata = np.loadtxt('some_Pk.dat',unpack=True)
    >>>Pmm.Pfit(kdata,Pdata)
    >>>Pmm_new = Pmm.Pfit(kdata,Pdata)

    """
    #FOR NOW will only support single z, but can come back to this later
    def __init__(self,cosmo,z,klin,plin):
        self.cosmo = cosmo
        self.z = z
        self.plin = loginterp(klin,plin) #interpolator
        #cache the necessary cosmo distance quantities for DS and shear (bin-independent)
        self.chi_z = cosmo.comoving_distance
        self.z_chi = ius(chi,z)
        self.dchidz = ...
        self.Dz = cosmo.comoving_transverse_distance
