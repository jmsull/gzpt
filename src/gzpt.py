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
    __init__

    Examples
    --------
    >>> k,plin = np.loatxt("my_linear_power_file.txt",unpack=True)
    >>> z = 0.5
    >>> model = gzpt.hzpt(k,plin,z)
    >>> mm = matter.Correlator(model,nmax=1)
    >>> gg = tracers.AutoCorrelator(model,nmax=1)
    >>> gm = tracers.CrossCorrelator(model,nmax=2)

    #3D stats
    r = np.logspace(-1,3)
    ximm = mm.Xi()(r)
    Pgg = gg.Power()(k)

    #Projected Stats
    >>> z_source = 1
    >>> DeltaSigma_gm = gm.Delta_Sigma(r,z_source) #using default top-hat pi-bins

    #Analytic gradients (of non-cosmology parameters)
    >>> wgg,grad_wgg = gg.wp(r,wantGrad=True)

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
