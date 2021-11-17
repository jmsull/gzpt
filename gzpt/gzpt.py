#tie stuff together around the hzpt object
import numpy as np
import warnings
from gzpt.zel import *
from scipy.interpolate import InterpolatedUnivariateSpline as ius
import os.path as path
import warnings

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

    Example
    --------
    >>> k,plin = np.loadtxt("my_linear_power_file.txt",unpack=True)
    >>> z = 0.5
    >>> model = gzpt.hzpt(k,plin)
    >>> mm = matter.Correlator(params, model)
    >>> gg = tracers.AutoCorrelator(params,model)
    >>> gm = tracers.CrossCorrelator(params,model)

    #3D stats
    r = np.logspace(-1,3)
    ximm = mm.Xi()(r)
    k = np.logspace(-3,1)
    Pgg = gg.Power()(k)

    """
    #For now will only support single z, but can come back to this later
    def __init__(self,klin,plin,config=True):
        self.plin = loginterp(klin,plin) #interpolator

        #compute ZA
        self.cleft = CLEFT(klin,plin)
        self.cleft.make_ptable()
        self.kza,pza = self.cleft.pktable.T #evaluated at klin
        self.P_zel = loginterp(self.kza,pza) #callable
        self.config = config
        if(self.config):
            self.rxi = np.logspace(-1,3,4000) #matching up with the SBT in zel
            xiza = self.cleft.compute_xi_real(self.rxi)
            self.Xi_zel = loginterp(self.rxi,xiza) #callable
        else:
            warnings.warn("Configuration space models will not work if config flag is off!",
                           RuntimeWarning)

    def update_redshift(self,Dz):
        self.cleft.make_ptable(Dz=Dz)
        self.P_zel = loginterp(*self.cleft.pktable.T)
        if(self.config):
            self.Xi_zel = loginterp(self.rxi,self.cleft.compute_xi_real(self.rxi))
