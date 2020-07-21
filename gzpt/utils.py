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

    """warning: using EdS growth"""
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

#Profiles
def rho_NFW(r,M,z,conc_relation=c_duffy,mdef='vir'):
    c = conc_relation(M,z,mdef=mdef)
    rS =  rDelta(M,z,mdef=mdef)/c

    def form(r):
        return ((r/rS)*(1 + r/rS)**2)**-1
    rint = np.logspace(-3,np.log10(rDelta(M,z,mdef=mdef)))
    norm = Delta(z,mdef)*M/(4*np.pi) * (np.trapz(form(rint)*Delta(z,mdef)*rint**2,rint))**-1#M * (4*np.pi*rS**3 *(np.log(1+c) - c/(1+c)))**-1

    return norm*form(r)

def rho_DK14ext(r,M,z,cosmo=Planck15):
    """Modified form of Deimer + Kravtsov 2014 - use NFW instead of Einasto for inner profile, choose fixed paramters for outer profile.
    Goal is simply qualitative inclusion of the outer halo mass transition."""

    ftrans = (1 + .2*(r/rDelta(M,z,mdef='200m'))**4 )**-(nusq(M,z)**(1/2))
    print("warning: normalization for DK14 is not correct - b_e is fixed")
    b_e = 1.5 #not 2halo adjusted or scaling appropriately with redshift
    s_e = 1.5
    outer = cosmo.rho_m(z)*1e10 * (b_e *(r/(5*rDelta(M,z,mdef='200m')))**-s_e +1)

    return rho_NFW(r,M,z,mdef='200m')*ftrans +outer


def rho_BCM(r,M,z,cosmo=Planck15):
    """Profile with baryonic corrections of Schneider ++ 2019, 2015 using a particular choice of baryon paramters.
    Goal is simply qualitative inclusion of baryonic effects."""
    print("Watch out - this profile is defined in terms of r200c, make sure you convert (because noting else is defined in 200c)!")
    #using "true values" of Schnieder et al.
    eta_c = 0.6
    eta_s = .32
    mu = .21
    M_c = 10**13.8
    theta_ej = 4
    beta = 3-(M_c/M)**mu

    f_star = 0.09*(M/2.5e11)**-eta_s
    f_cga = 0.09*(M/2.5e11)**-eta_c
    f_sga = f_star-f_cga
    f_gas = cosmo.Omega_b(z)/cosmo.Omega_m(z) - f_star

    r200c = rDelta(M,z,mdef='200c')
    r_co = 0.1*r200c
    r_ej = theta_ej*r200c
    Rh = 0.015*r200c

    norm1 = M/(np.pi**(3/2) * 4. * Rh)

    term1 = norm1 * (f_cga/r**2 * np.exp(-(r/Rh/2.)**2))

    term2 = (cosmo.Omega_cdm(z)/cosmo.Omega_m(z)+ f_sga) *rho_NFW(r,M,z)

    def term3(r):
        return (1/ ((1 + r/r_co)**(beta) * (1+(r/r_ej)**2)**(.5*(7-beta))))
    rint = np.logspace(-2,r200c)
    norm3 = f_gas * M/(4.*np.pi) * (np.trapz(term3(rint),rint))**-1

    term3 = norm3*term3(r)

    return term1 + term2 + term3

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
    # def mass_cutoff():
    #     """"Mass conserving cutoff of Schmidt 2016"""
    #
    # def consistency(): #don't really need if no bias...
    nu = nusq(M,z)**(1/2) #peak height should always use Lagrangian sigma

    if(choice=='ST'):
        #print("warning, these limits are made up - check them!")
        print("Using ST, assumes Mvir")
        #Mmin = 1e10
        #Mmax = 1e16
        fnu = ST(nu,**fnu_params) #nu**1/2?
    elif(choice=='Tinker'):
        print("Using Tinker, assumes M200m")
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


#HZPT integrals over profiles
def hzpt_integrals(z,profile=rho_NFW,mdef='vir',cosmo=Planck15,rMin=1e-2,rMax=10,num_pts=100,integ_m=1e13,nmax=2,Mmin=10**10.5,Mmax=10**15.5,hmf_options={}):
    """
    HMF - callable dndM mass function
    z - redshift, float
    """

    def I0(r,M):
        return 4*np.pi/M * r**2 *profile(r,M,z)

    def I1(r,M):
        return 4*np.pi/M/6 * r**4 *profile(r,M,z)

    def I2(r,M):
        return 4*np.pi/M/120 * r**6 *profile(r,M,z)

    rr = np.logspace(np.log10(rMin),np.log10(rMax),num_pts)
    m = np.logspace(np.log10(Mmin),np.log10(Mmax),num_pts)
    mbrmsq = (m/(cosmo.rho_m(z)*1e10))**2

    #In this function, for all integrals we take the postiive sqrt/4thrt soln since it is the only one that makes sense
    #It will not matter if the "bare" R_n(h) is positive or negaive because it only enters as squared or ^4 - we don't permit imaginary R

    integrals,integrands=[],[]
    @np.vectorize
    def F0(M):
         rr = np.logspace(np.log10(rMin),np.log10(rDelta(M,z,mdef=mdef)),num_pts)
         return np.trapz(I0(rr,M),x=rr)
    integrands.append(I0(rr,integ_m))
    A0 = np.trapz(dndM(m,z,**hmf_options) * mbrmsq * F0(m)**2,x=m)
    integrals.append(A0)
    def pade(k):
        return A0

    if(nmax>=1):
        @np.vectorize
        def F1(M):
            rr = np.logspace(np.log10(rMin),np.log10(rDelta(M,z,mdef=mdef)),num_pts)
            return np.trapz(I1(rr,M),x=rr)
        integrands.append(I1(rr,integ_m))
        R1bar = (np.trapz(dndM(m,z,**hmf_options) * mbrmsq * 2.*F1(m)*F0(m),x=m)/A0)**(1/2) #this term has a minus sign in front of it!
        integrals.append(R1bar)
        R1h0 = R1bar
        def pade(k):
            pade = 1./(1. + k**2 * R1h0**2)

        if(nmax==2):
            @np.vectorize
            def F2(M):
                rr = np.logspace(np.log10(rMin),np.log10(rDelta(M,z,mdef=mdef)),num_pts)
                return np.trapz(I2(rr,M),x=rr)
            integrands.append(I2(rr,integ_m))
            R2bar = (np.trapz(dndM(m,z,**hmf_options) * mbrmsq * (F1(m)**2 + 2.*F0(m)*F2(m)),x=m)/A0)**(1/4)
            integrals.append(R2bar)
            Q = R1bar**4 - R2bar**4
            R1h = (R1bar**2 * R2bar**4 /Q)**(1/2)
            R1 = (R1bar**2 *(R1bar**4 - 2.*R2bar**4) / Q)**(1/2)
            R2h = (R2bar**8 / Q)**(1/4)

            def pade(k):
                pade = (1. - k**2 * R1**2)/(1. + k**2 * R1h**2 + k**4 * R2h**4)

    if(integ_m is not None):
        return integrals,pade,integrands,rr
    else:
        return integrals,pade


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

'''Check consistency constraint!'''
