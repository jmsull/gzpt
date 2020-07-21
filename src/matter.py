#standard dm HZPT
from gzpt import hzpt
import zel,bb
import numpy as np
from scipy.special import hyp0f1,gamma
from numpy import inf


class Correlator(hzpt):
    '''hzpt provides cosmology, plin, z
    '''
    def __init__(self,params,hzpt,nmax):
        #Set maximum Pade expansion order
        #self.nmax = nmax
        self.z = hzpt.z #redshift

        #Power laws in s8 from pyRSD P_mm may want to update these
        A0_init,R_init =
        R1h_init =
        R1sq_init,R2h_init =

        names = np.array(['A0','R','R1h','R1sq','R2h'])

        self.params = params
        self.nmax=len(params)

        if(self.nmax==0):
            self.params = np.array([A0_init,R_init]) #A0, R
            self.pnames = names[:2]
        elif(self.nmax==1):
            self.params = np.array([A0_init,R_init,R1h_init ]) #A0, R, R1h
            self.pnames = names[:3]
        elif(self.nmax==2):
            self.params = np.array([A0_init,R_init,R1h_init, R1sq_init,R2h_init]) #A0, R, R1h, R1sq, R2h
            self.pnames = names
        elif(self.nmax==3):
            self.params = np.array([A0_init,R_init,R1h_init, R1sq_init,R2h_init, R2sq_init,R3h_init]) #for P only
        else:
            raise(NotImplementedError)

        #replace others with this after is working - currently only using for fits
        self.pdict = dict(zip(self.pnames,self.params))

    def Power(self):
        '''Returns callable that incorporates the current parameters'''
        def Pk(k):
            return self.P_zel(k,self.z) + bb.PBB(k,params,nmax=self.nmax)
        return Pk

    def grad_Power0(self,k,A0,R):
        params = np.array([A0,R])
        def gradP(k):
            return bb.PBB(k,params,nmax=self.nmax,wantGrad=True)[1]
        return gradP(k)

    def grad_Power1(self,k,A0,R,R1h):
        params = np.array([A0,R,R1h])
        def gradP(k):
            return bb.PBB(k,params,nmax=self.nmax,wantGrad=True)[1]
        return gradP(k)

    def grad_Power2(self,k,A0,R,R1h,R1sq,R2h):
        params = np.array([A0,R,R1h,R1sq,R2h])
        def gradP(k):
            return bb.PBB(k,params,nmax=self.nmax,wantGrad=True)[1]
        return gradP(k)

    def Xi(self):
        '''Returns callable that incorporates the current parameters'''
        if(len(params)<1):params=self.params
        def xi(r):
            return self.Xi_zel(r,self.z) + bb.XiBB(r,params,nmax=self.nmax)
        return xi

    def grad_Xi0(self,r,A0,R):
        params = np.array([A0,R])
        def gradX(r):
            bb_grad = bb.XiBB(r,params,nmax=self.nmax,wantGrad=True)[1]
            return bb_grad
            #return np.atleast_2d(r**2).T * bb_grad
        return gradX(r)

    def grad_Xi1(self,r,A0,R,R1h):
        params = np.array([A0,R,R1h])
        def gradX(r):
            bb_grad = bb.XiBB(r,params,nmax=self.nmax,wantGrad=True)[1]
            return bb_grad
            #return np.atleast_2d(r**2).T * bb_grad
        return gradX(r)

    def grad_Xi2(self,r,A0,R,R1h,R1sq,R2h):
        params = np.array([A0,R,R1h,R1sq,R2h])
        def gradX(r):
            bb_grad = bb.XiBB(r,params,nmax=self.nmax,wantGrad=True)[1]
            return bb_grad
            #return np.atleast_2d(r**2).T * bb_grad
        return gradX(r)

    def wp(self,r,pi_bins=np.linspace(0,100,10,endpoint=False)):
        "FIXME: Lazy copy from AutoCorrelator - should make this function accessible by both. - general correlator class..."
        """Projected correlation function for mm (convergence).
        Input:
        r - array
        pi_bins - array (of size that divides evenly into r) to projected
        Output:
        rp, projected radius (over pi)
        wp, projected correlation function
        """
        #Almost as in corrfunc code of Sinha & Garrison
        xi = self.Xi()(r)
        dpi = pi_bins[1]-pi_bins[0]
        wp = np.zeros(int(len(xi)/len(pi_bins)))
        #sum over los direction in each bin
        for i in range(len(wp)-1):
            wp[i] = 2* dpi * np.sum(xi[i*len(pi_bins):(i+1)*len(pi_bins)])
        rp = r[::len(pi_bins)]
        return rp,wp

    def grad_wp():

        return None

    def chi_z(self,z,useNorad=True):
        '''Set cached interpolators for chi(z) using fast approximation if desired (20x speedup)'''
        #may want to put this in HZPT object...
        if(useNorad):
            norad = self.hzpt.cosmo.clone(Tcmb0=0.)
            use_cosmo = norad
            chi = use_cosmo.comoving_distance(z)
        else:
            use_cosmo = self.hzpt.cosmo
            chi = use_cosmo.comoving_distance(z)

        self.chi_z = ius(z,chi,k=1,ext=0)
        self.dchi_dz = self.chi_z.derivative()#ius(z,3e5 / use_cosmo.H(z)) #have not checked norad approx for H, use dzdchi = 1/dchidz
        self.z_chi = ius(chi,z,k-1,ext=0)

        return None

    def shear_xi(self,theta,chi_max,
                 z_i,nz_i,m_i,
                 z_j=None,nz_j=None,m_j=None,
                 wantPkappa=False):
        """
        Shear correlation functions
        Input:
        theta-array
        pi_bins -array of size that divides evenly into r
        nz_i,nz_j -n(z) arrays for possibly two different bins
        m_i, m_j - multiplicative bias for two different redshift bins, if autocorrelation, use m_i
        Ouput:
        rp, projected radius
        shear_+, J0 shear
        shear_-, J4 shear
        """
        #Bessel functions
        def J0(x): return hyp0f1(0+1,-x**2 /4.)/gamma(0+1)
        def J4(x): return (x/2.)**4 * hyp0f1(4+1,-x**2 /4.)/gamma(4+1)

        if(m_j==None or nz_j==None):
            m_j=m_i
            nz_j=nz_i

        #lens efficiency, DES paper, assume chimin=0 and linspace is appropriate (may not be)
        chi_prime = np.linspace(0.,chi_max,num_chi_pts)
        chi = np.copy(chi_prime) #to stick with notation


        def nchi(chi,z,nz):
            """Normalized n(chi) from n(z)"""
            nbar = np.trapz(nz,x=z)
            nchi = self.chi_z(z),nz/self.dchi_dz(z)/nbar
            return nchi

        def q_eff(z,nz,chi,chi_prime=chi_prime): return chi*(1+z(chi))*np.trapz(nchi(chi_prime,z,nz) * (chi_prime-chi)/chi_prime,x=chi_prime)

        q_i = q_eff(z,nz_i,chi,chi_prime)
        if(nz_j==nz_i):
            q_j = q_i
        else:
            q_j = q_eff(z_j,nz_j,chi,chi_prime)

        #shared shear
        pre_fac = (1.+m_i)*(1+m_j)/(2.*np.pi)
        def inner_integrand(chi,ell): return ell*q_i*q_j * chi**-2 * self.Power()((ell+.5)/chi, self.z_chi(chi)) #NTLO limber
        def outer_integrand(ell): return np.trapz(inner_integrand(chi,ell),x=chi) #shared inner integrand

        #xi_+
        def shear_plus_theta(theta): return np.trapz(J0(ell*theta)*outer_integrand(ell),x=ell)
        #xi_-
        def shear_minus_theta(theta): return np.trapz(J4(ell*theta)*outer_integrand(ell),x=ell)

        #return at all input thetas
        if(not wantPkappa):
            return shear_plus_theta(theta),shear_minus_theta(theta)
        else:
            #P_kappa(ell)
            H0 = cosmo.h*100.
            pre_fac = 3*H0**2 *cosmo.Omega_m / (2.*3e5**2)
            Pkappa_ell = pre_fac**2 * np.trapz(q_i*q_j*self.Power()(ell+0.5)/chi,x=chi)
            return shear_plus_theta(theta),shear_minus_theta(theta),Pkappa_ell

        #not sure if trapz will be enough here
