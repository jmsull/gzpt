#Misc. halo model functions
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
        #print("Using linear power - interpolating planck")
        kk,Pkk = np.loadtxt('/Users/jsull/Cosmology_Codes/flowpm/flowpm/data/Planck15_a1p00.txt',unpack=True)#LinearPower(cosmo,z) #too slow, running class each time
        def P(k):
            return np.interp(k,kk,Pkk)

    k = np.logspace(np.log10(kmin),np.log10(kmax),num_pts)

    """using EdS growth"""
    def I(k):
        I = k**2 * P(k) * np.abs(W_TH(k,rDelta(M,z,mdef)))**2
        return I
    Ig = growth*np.sqrt((1/(2*np.pi**2))* np.trapz(I(k),x=k))# quad(I,kmin,kmax)[0])

    return Ig

def Mstar(z,tol=1e-1,num_pts=100,lMmin=10,lMmax=14):
    "inverse of sigma(M,z,mdef)"
    print("assuming Planck cosmology - should really put all cosmology-depenent functions in a class - simga, nu, mstar, etc.?")
    Minterp = np.logspace(lMmin,lMmax,num_pts)
    nus = nusq(Minterp,z)
    Mstar = Minterp[np.median(np.argwhere(abs(nus-1)<tol)).astype(int)]
    return Mstar

def nusq(M,z):
    delta_c = 1.686
    nu = (delta_c/sigma(M,z,'Lag'))**2 #peak height always in Lag
    return nu

#Concentration
def c_duffy(M,z,mdef='vir'):
    "Duffy 2008 concentration"
    #add 200c
    if(mdef=='vir'):
        a = 7.85
        b = -0.081
        c = -0.71
    elif(mdef=='200m'):
        a = 10.14
        b = -0.081
        c = -1.01
    else:
        raise NotImplementedError("Sorry - mass definition not implemented!")
    M0 = 2e12
    return a * (M/M0)**b * (1+z)**c

#Mass functions
def ST(nu,
       p=0.3,
       q=0.707,
       A=0.21616):
    "f(nu) Sheth-Tormen 1999 - I guess nu includes all z dependence? may need to refit? Should use this for Mvir/Mfof"

    return A*(1 + (q*nu**2)**(-p))*np.exp(-q*nu**2 /2)

def Tinker(nu,z):
    ''''Tinker 2010 eqn. 8 - Note - should only use with R200m!'''
    #z=0, values for Delta=200
    alpha = 0.368
    beta_0 = 0.589
    phi_0 = -0.729
    eta_0 = -0.243
    gamma_0 = 0.864

    #z evol
    beta = beta_0*(1+z)**0.2
    phi = phi_0*(1+z)**-0.08
    eta = eta_0*(1+z)**0.27
    gamma = gamma_0*(1+z)**-0.01

    fnu =  alpha * (1+ (beta * nu)**(-2*phi) ) * nu**(2*eta) *np.exp(-gamma * nu**2 /2)

    return fnu


def dndM(M,z,choice='Tinker',num_pts=100,cosmo=Planck15,fnu_params={}): #Make a class
    nu = nusq(M,z)**(1/2) #peak height should always use Lagrangian sigma

    if(choice=='ST'):
        #print("Using ST, assumes Mvir")
        #Mmin = 1e10
        #Mmax = 1e16
        fnu = ST(nu,**fnu_params) #nu**1/2?
    elif(choice=='Tinker'):
        #print("Using Tinker, assumes M200m")
        #print("warning: using Planck15, cosmology should probably be WMAP9?")
        #calibrated range of Tinker - I have checked this integrates to 1
        #Mmin = 10**10.2
        #Mmax = 10**15.5
        fnu = Tinker(nu,z)
    else:
        raise ValueError("Mass function choice not supported.")



    def dinvsigdM_prime(lM,z):
        """Finite differences gradient for inv sigma, should be no problem since v smooth"""
        interp_lM = np.linspace(10.2,15.5,num_pts)
        #return np.interp(lM,interp_lM,np.gradient(np.log(1/(sigma(10**interp_lM,z,mdef='200m'))),interp_lM)) #confusingly, the sigma here is sigma_m not sigma_lag
        return np.interp(lM,interp_lM,np.gradient(np.log(1/(sigma(10**interp_lM,z,mdef='Lag'))),interp_lM)) #no it would be insane to make sigma mean two separate things in one line - both should be Lag I think


    dndM = fnu * (1e10*cosmo.rho_m(z)/M) * dinvsigdM_prime(np.log10(M),z)#/M


    return dndM

def b_Tinker(M,z):
    '''Tinker 2010 halo bias - to get initial guess for b1'''
    nu = np.sqrt(nusq(M,z))

    Delta = 200
    delta_c = 1.6186
    y = np.log10(Delta)
    t = y*np.exp(-(4/y)**4)

    A = 1. + .24*t
    B = 0.183
    C = 0.019 + 0.107*y +0.19*t/y
    a = 0.44*y - 0.88
    b = 1.5
    c = 2.4

    return 1 - A*(nu**a / (nu**a + delta_c**a)) + B*nu**b + C*nu**c
