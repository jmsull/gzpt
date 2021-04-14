#Misc. functions
from nbodykit.cosmology import Planck15
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

#true utils
def W_TH(k,R):
    '''Top hat window
    Input: wavenumber k - arraylike
    position r - arraylike
    option (k,r) - for Fourier or configuration space top hat
    R - the filtering scale
    '''
    x=k*R
    return (3/x**3)*(np.sin(x) - x*np.cos(x))


def W_TH_real(r,R):
    '''Top hat window
    Input:
    position r - arraylike
    R - the filtering scale
    '''
    V = 4*np.pi/3 *R**3
    if(len(r)>0):
        indicator = np.ones(r.shape)
        indicator[r>R] =0
        return indicator/V
    else:
        if(r>R):
            return 0.
        else:
            return 1/V

def Nk(k,L=3200.):
    """Number of k modes for a given box size. Default is cc box size."""
    kf = 2.*np.pi/L
    vf = (kf)**3
    dk = kf #fftpower uses fundamental mode for dk by default
    Nk = (4*np.pi*k**2 * dk)/vf
    return Nk

def match(rlow,xlow,rhigh,xhigh,mp=10,extrap=1,npts=100,bare=False):
    """For plotting 2pcfs - interpolate small-scale pair counts and large scale FFT grid 2PCF"""
    """Output is r^2 xi """
    rlow,xlow = rlow[rlow>0],xlow[rlow>0] #check if zero because sometimes that happens for nbodykit 2pcf
    m = mp #match_point
    rconc,xconc = np.concatenate([rlow[rlow<m],rhigh[rhigh>=m]]),np.concatenate([xlow[rlow<m],xhigh[rhigh>=m]])
    r = np.logspace(np.log10(rlow.min()),np.log10(rhigh.max()),npts)
    s = ius(rconc,rconc**2 * xconc,ext=extrap)(r)
    if(bare): #return multiplied by r^2 by default
        s = s/r**2
    return r,s

def Delta(z,mdef,cosmo=Planck15):
    if(mdef=='vir'):
        '''Bryan + Norman 1998 fit'''
        xv = 1- cosmo.Omega_m(z)
        return ((18*np.pi**2 - 82*xv -39*xv**2)/(1-xv) * cosmo.rho_crit(z)*1e10)
    elif(mdef=='200m'):
        return 200*cosmo.rho_m(z)*1e10
    elif(mdef=='200c'):
        return 200*cosmo.rho_crit(z)*1e10
    elif(mdef=='Lag'):
        return cosmo.rho_m(z)*1e10
    elif(mdef=='exc'):
        "Approx. Baldauf++ 13 fitting value for z=0"
        return 30*cosmo.rho_m(z)*1e10
    else:
        print("Mass definition not avaliable!")
        raise ValueError

def rDelta(M,z,mdef='vir',cosmo=Planck15):
    "Choosing vir since closest to what M_FoF finds with ll=0.2 (White 2000 Table 1)"
    return ((3/(4*np.pi*Delta(z,mdef,cosmo)))*M)**(1/3)

def mDelta(r,z,mdef='vir',cosmo=Planck15):
    return 4/3 *np.pi*Delta(z,mdef,cosmo)*r**3

@np.vectorize
def sigma(M,z,mdef,P=None,kmin=1e-5,kmax=1e2,num_pts=100,cosmo=Planck15):
    '''
    Get sigma from P using trapezoidal rule
    Input:
    M: Mass defining smoothing scale
    z: redshift
    mdef: choice of mass definition for smoothing window
    optional
    P: Power spectrum callable, if none use linear
    kmin: lower integration range
    kmax: upper integration range
    num_pts: points to use in integration
    '''
    growth = (1/(1+z))
    if P is None:
        kk,Pkk = np.loadtxt('/Users/jsull/Cosmology_Codes/flowpm/flowpm/data/Planck15_a1p00.txt',unpack=True)
        def P(k):
            return np.interp(k,kk,Pkk)

    k = np.logspace(np.log10(kmin),np.log10(kmax),num_pts)

    """using EdS growth"""
    def I(k):
        I = k**2 * P(k) * np.abs(W_TH(k,rDelta(M,z,mdef)))**2
        return I
    Ig = growth*np.sqrt((1/(2*np.pi**2))* np.trapz(I(k),x=k))

    return Ig
