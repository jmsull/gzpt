#tie stuff together around the hzpt object
import numpy as np
import warnings
from gzpt.zel import *
from scipy.interpolate import InterpolatedUnivariateSpline as ius
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
    >>> k,plin = np.loadtxt("my_linear_power_file.txt",unpack=True)
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

    #Analytic gradients (of non-cosmology hzpt parameters)
    >>> wgg,grad_wgg = gg.wp(r,wantGrad=True)

    """
    #FOR NOW: will only support single z, but can come back to this later
    def __init__(self,klin,plin):
        self.plin = loginterp(klin,plin) #interpolator

        #compute ZA
        self.cleft = CLEFT(klin,plin)
        self.cleft.make_ptable()
        kt,plir,pza = self.cleft.pktable.T #evaluated at klin
        self.P_zel = loginterp(kt,pza) #callable
        self.P_lir = loginterp(kt,plir)
        rxi = np.logspace(-1,3,4000) #matching up with the SBT in zel
        xiza = self.cleft.compute_xi_real(rxi)
        self.Xi_zel = loginterp(rxi,xiza) #callable
