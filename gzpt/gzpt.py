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

    #Analytic gradients (of non-cosmology hzpt camb_mnu0.15_transfer_out_z0.0_rescaled.datparameters)
    >>> wgg,grad_wgg = gg.wp(r,wantGrad=True)

    """
    #FOR NOW will only support single z, but can come back to this later
    def __init__(self,klin,plin,z):
        #self.cosmo = cosmo
        self.z = z
        self.plin = loginterp(klin,plin) #interpolator
        #cache the necessary cosmo distance quantities for DS and shear (bin-independent) - later
        # self.chi_z = cosmo.comoving_distance
        # self.z_chi = ius(chi,z)
        # self.dchidz = ...
        # self.Dz = cosmo.comoving_transverse_distance

        #compute ZA
        self.cleft = CLEFT(klin,plin)
        self.cleft.make_ptable()
        _,pza = self.cleft.pktable.T #evaluated at klin
        self.P_zel = loginterp(klin,pza) #callable
        rxi = np.logspace(-1,3,4000) #matching up with the SBT in zel
        xiza = self.cleft.compute_xi_real(rxi)
        self.Xi_zel = loginterp(rxi,xiza) #callable
