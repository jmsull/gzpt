#standard dm HZPT
from gzpt import hzpt
from gzpt.bb import *
import numpy as np
from scipy.special import hyp0f1,gamma
from numpy import inf


class Correlator(hzpt):
    '''hzpt provides cosmology, plin, z
    '''
    def __init__(self,params,hzpt):
        '''
        Parameters
        ----------
        params : list (float)
            List of current hzpt parameter values.
        hzpt : hzpt
            Hzpt base class that holds the linear, zeldovich, and z information.
        '''
        #should inherit self.z = hzpt.z #redshift
        self.params = params
        self.nmax=(len(self.params)-1)//2
        assert self.nmax<=3
        #probably a better way to do this...
        self.hzpt = hzpt

    def Power(self,wantGrad=False):
        '''
        Parameters
        ----------
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns
        ---------
        callable
            Power spectrum in k (h/Mpc)
        '''
        def Pk(k):
            if(wantGrad):
                bb,bbgrad = PBB(k,self.params,nmax=self.nmax,wantGrad=True)
                return self.hzpt.P_zel(k) + bb, bbgrad
            else: return self.hzpt.P_zel(k) + PBB(k,self.params,nmax=self.nmax)
        return Pk

    def Xi(self,wantGrad=False):
        '''
        Parameters
        ----------
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns
        ---------
        callable
            2pcf function of r (Mpc/h)
        '''
        def xi(r):
            if(wantGrad):
                bb,bbgrad = XiBB(r,self.params,nmax=self.nmax,wantGrad=True)
                return self.hzpt.Xi_zel(r) + bb, bbgrad
            else: return self.hzpt.Xi_zel(r) + XiBB(r,self.params,nmax=self.nmax)
        return xi

    def wp(self,r,pi_bins=np.linspace(0,100,10,endpoint=False),wantGrad=False):
        "FIXME: Lazy copy from AutoCorrelator - should make this function accessible by both. - general correlator class..."
        """Projected correlation function.
        Parameters:
        -----------
        r: array (float)
            3D abscissa values for xi
        pi_bins: array (float)
            Array must be of size that divides evenly into r - projection window, tophat for now
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns:
        ----------
        (array (float), array (float), [array (float)])
            projected radius rp, projected correlation function wp, gradient if wanted
        """
        #Almost as in corrfunc code of Sinha & Garrison
        dpi = pi_bins[1]-pi_bins[0]
        if(wantGrad):
            xi,grad_xi = self.Xi(wantGrad=wantGrad)(r)
        else:
            xi = self.Xi(wantGrad=wantGrad)(r)
        wp = np.zeros(int(len(xi)/len(pi_bins)))
        if(wantGrad): wp_grad=np.zeros((len(wp),self.nmax))

        #sum over los direction in each bin
        for i in range(len(wp)-1):
            wp[i] = 2* dpi * np.sum(xi[i*len(pi_bins):(i+1)*len(pi_bins)])
            if(wantGrad): wp_grad[i] = 2* dpi * np.sum(grad_xi[i*len(pi_bins):(i+1)*len(pi_bins)])
        rp = r[::len(pi_bins)]
        #TODO: All of this branching is probably bad - come back and fix this
        if(wantGrad):
            return rp,wp,wp_grad
        else:
            return rp,wp

    #-----------------------Out of scope of first release, will move
    # def chi_z(self,z,useNorad=True):
    #     '''Set cached interpolators for chi(z) using fast approximation if desired (20x speedup)'''
    #     #may want to put this in HZPT object...
    #     if(useNorad):
    #         norad = self.hzpt.cosmo.clone(Tcmb0=0.)
    #         use_cosmo = norad
    #         chi = use_cosmo.comoving_distance(z)
    #     else:
    #         use_cosmo = self.hzpt.cosmo
    #         chi = use_cosmo.comoving_distance(z)
    #
    #     self.chi_z = ius(z,chi,k=1,ext=0)
    #     self.dchi_dz = self.chi_z.derivative()#ius(z,3e5 / use_cosmo.H(z)) #have not checked norad approx for H, use dzdchi = 1/dchidz
    #     self.z_chi = ius(chi,z,k-1,ext=0)
    #
    #     return None
    #
    # def shear_xi(self,theta,chi_max,
    #              z_i,nz_i,m_i,
    #              z_j=None,nz_j=None,m_j=None,
    #              wantPkappa=False):
    #     """
    #     Shear correlation functions
    #     Input:
    #     theta-array
    #     pi_bins -array of size that divides evenly into r
    #     nz_i,nz_j -n(z) arrays for possibly two different bins
    #     m_i, m_j - multiplicative bias for two different redshift bins, if autocorrelation, use m_i
    #     Ouput:
    #     rp, projected radius
    #     shear_+, J0 shear
    #     shear_-, J4 shear
    #     """
    #     #Bessel functions
    #     def J0(x): return hyp0f1(0+1,-x**2 /4.)/gamma(0+1)
    #     def J4(x): return (x/2.)**4 * hyp0f1(4+1,-x**2 /4.)/gamma(4+1)
    #
    #     if(m_j==None or nz_j==None):
    #         m_j=m_i
    #         nz_j=nz_i
    #
    #     #lens efficiency, DES paper, assume chimin=0 and linspace is appropriate (may not be)
    #     chi_prime = np.linspace(0.,chi_max,num_chi_pts)
    #     chi = np.copy(chi_prime) #to stick with notation
    #
    #
    #     def nchi(chi,z,nz):
    #         """Normalized n(chi) from n(z)"""
    #         nbar = np.trapz(nz,x=z)
    #         nchi = self.chi_z(z),nz/self.dchi_dz(z)/nbar
    #         return nchi
    #
    #     def q_eff(z,nz,chi,chi_prime=chi_prime): return chi*(1+z(chi))*np.trapz(nchi(chi_prime,z,nz) * (chi_prime-chi)/chi_prime,x=chi_prime)
    #
    #     q_i = q_eff(z,nz_i,chi,chi_prime)
    #     if(nz_j==nz_i):
    #         q_j = q_i
    #     else:
    #         q_j = q_eff(z_j,nz_j,chi,chi_prime)
    #
    #     #shared shear
    #     pre_fac = (1.+m_i)*(1+m_j)/(2.*np.pi)
    #     def inner_integrand(chi,ell): return ell*q_i*q_j * chi**-2 * self.Power()((ell+.5)/chi, self.z_chi(chi)) #NTLO limber
    #     def outer_integrand(ell): return np.trapz(inner_integrand(chi,ell),x=chi) #shared inner integrand
    #
    #     #xi_+
    #     def shear_plus_theta(theta): return np.trapz(J0(ell*theta)*outer_integrand(ell),x=ell)
    #     #xi_-
    #     def shear_minus_theta(theta): return np.trapz(J4(ell*theta)*outer_integrand(ell),x=ell)
    #
    #     #return at all input thetas
    #     if(not wantPkappa):
    #         return shear_plus_theta(theta),shear_minus_theta(theta)
    #     else:
    #         #P_kappa(ell)
    #         H0 = cosmo.h*100.
    #         pre_fac = 3*H0**2 *cosmo.Omega_m / (2.*3e5**2)
    #         Pkappa_ell = pre_fac**2 * np.trapz(q_i*q_j*self.Power()(ell+0.5)/chi,x=chi)
    #         return shear_plus_theta(theta),shear_minus_theta(theta),Pkappa_ell
    #
    #     #not sure if trapz will be enough here
