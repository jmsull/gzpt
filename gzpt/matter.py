#standard dm HZPT
from gzpt import hzpt
from gzpt.bb import *
import numpy as np
from scipy.special import hyp0f1,gamma
from numpy import inf


class Correlator(hzpt):
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
        if(wantGrad): wp_grad=np.zeros((len(wp),len(self.params)))

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
