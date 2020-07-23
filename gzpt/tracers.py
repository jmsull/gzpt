'''Real-space tracer/galaxy/halo model'''
from gzpt import hzpt
from gzpt.bb import *
import numpy as np
from numpy import inf
from scipy.special import erf
from scipy.interpolate import InterpolatedUnivariateSpline as ius

class AutoCorrelator(hzpt):
    '''hzpt object provides:
    '''
    def __init__(self,params,hzpt,useExc=True):
        '''
        Parameters
        ----------
        params : list (float)
            List of current hzpt parameter values.
        hzpt : hzpt
            Hzpt base class that holds the linear, zeldovich, and z information.
        useExc : boolean,optional
            Whether or not to use exclusion in the model.
        '''
        self.params = params
        self.nmax = min((len(self.params[2:])-1)//2,2)
        self.useExc = useExc
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
        if(not self.useExc):
            nbar,b1,pparams = self.params[0],self.params[1],self.params[2:]
        else:
            raise(NotImplementedError)
            print("Exclusion for power not implemented yet (and is not necessary for % accuracy).")
            nbar,b1,pparams = self.params[0],self.params[1],self.params[2:-2]#,params[-2:]

        def Pk(k):
            if(wantGrad):
                bb,bb_grad = PBB(k,pparams,nmax=self.nmax,wantGrad=True)
                p = 1/nbar + b1**2 * (self.hzpt.P_zel(k) + bb)
                nbar_grad = -(1/nbar**2) *np.ones(len(k))
                b1_grad = 2*(p- 1/nbar)/b1
                grad_p  = np.hstack([np.atleast_2d(nbar_grad).T,
                                     np.atleast_2d(b1_grad).T,
                                     b1**2 * bb_grad])
                return p,grad_p
            else:
                return p
        return Pk

    '''EXCLUSION'''

    def F_excl(self,r,R_excl,sigma_excl,wantGrad=False):
        '''
        Error function for the exclusion step as in Baldauf++ 2013 eqns (C2), (C4), with log10 as in main text
        Parameters
        ----------
        r : array (float)
            pts to evaluate xi
        R_excl : float
            Exclusion radius
        sigma_excl:
            Exclusion step width for error function
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns
        ---------
        array or (array,array)
            Exclusion kernel, and gradient, if asked for
        '''
        F = .5*(1 + erf(np.log10(r/R_excl)/(np.sqrt(2)*sigma_excl)))
        if(wantGrad):
            ln10 = np.log(10)
            exp = -np.exp(-np.log(r/R_excl)**2 / (2*sigma_excl**2 * ln10**2))
            denom = np.sqrt(2.*np.pi)* sigma_excl * ln10
            grad_Re = exp*denom/R_excl
            grad_sigmae = exp*denom*np.log(r/R_excl)/sigma_excl
            grad_F = [grad_Re,grad_sigmae]
            return F,grad_F
        else:
            return F

    def Xi(self,wantGrad=False):
        '''
        This function is a mess - come back and clean it up!
        Parameters
        ----------
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns
        ---------
        callable
            2pcf function of r (Mpc/h)
        '''
        if(self.useExc):
            if(self.nmax==1):
                b1,pparams,eparams = self.params[1],self.params[2:-2],self.params[-2:]
            elif(self.nmax==2):
                b1,pparams,eparams,ohparams = self.params[1],self.params[2:-4],self.params[-4:-2],self.params[-2:]
            R_excl,sigma_excl = eparams[0],eparams[1]
        else:
            b1,pparams,eparams = self.params[1],self.params[2:],None

        def xi(r):
            #Set exclusion
            if(self.useExc):
                R,sigma = eparams
                if(wantGrad):
                    exclusion,e_grad = self.F_excl(r,R,sigma,wantGrad=True)
                    grad_Re,grad_sigmae = e_grad
                else:
                    exclusion = self.F_excl(r,R,sigma)
            else:
                exclusion=1.

            if(self.nmax==2 and self.useExc):
                R_s_fixed=1e3
                if(wantGrad):
                    bb,bbgrad = XiBB(r,pparams,nmax=1,wantGrad=True)
                    xic = b1**2 * (self.hzpt.Xi_zel(r) + bb)
                    xi_baldauf_d = exclusion*(1. + xic) - 1.
                    b1_grad = 2.*xic/b1
                    bb_S, bbgrad_S = XiBB(r,[ohparams[0],R_s_fixed,ohparams[1]],nmax=1,wantGrad=True)
                    if(self.useExc):
                        return xi_baldauf_d + bb_S,np.hstack([np.atleast_2d(b1_grad).T,
                                                              b1**2 * bbgrad,
                                                              np.atleast_2d(grad_Re*(1. + xic)).T,
                                                              np.atleast_2d(grad_sigmae*(1. + xic)).T,
                                                              b1**2 * np.array([bbgrad_S[:,0],bbgrad_S[:,1]]).T
                                                              ])
                else:
                    xihzpt = b1**2 * (self.hzpt.Xi_zel(r)+ XiBB(r,pparams,nmax=1))
                    xi_baldauf_d = exclusion*(1+xihzpt)-1
                    return xi_baldauf_d +  XiBB(r,[ohparams[0],R_s_fixed,ohparams[1]],nmax=1) #set compensation R to existing value

            else:
                if(wantGrad):
                    bb,bbgrad = XiBB(r,pparams,nmax=self.nmax,wantGrad=True)
                    xic = b1**2 * (self.hzpt.Xi_zel(r) + bb)
                    xi_baldauf_d = exclusion*(1. + xic) - 1.
                    b1_grad = 2.*xic/b1
                    if(self.useExc):
                        return xi_baldauf_d,np.hstack([np.atleast_2d(b1_grad).T,
                                                       b1**2 * bbgrad,
                                                       np.atleast_2d(grad_Re*(1. + xic)).T,
                                                       np.atleast_2d(grad_sigmae*(1. + xic)).T
                                                       ])
                    else:
                        return xi_baldauf_d,np.hstack([np.atleast_2d(b1_grad).T,
                                                       b1**2 * bbgrad])
                else:
                    xihzpt = b1**2 * (self.hzpt.Xi_zel(r)+ XiBB(r,pparams,nmax=self.nmax))
                    xi_baldauf_d = exclusion*(1+xihzpt)-1
                    return xi_baldauf_d

        return xi

    '''Copying for now - will move projected statistics to another file later'''
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
        #fast check for bin spacing to return appropriate bin center values
        #r[0] may be 0
        btol=1e-2
        if(abs((r[2]-r[1])-(r[3]-r[2]))<=btol):
            #rp is currently of size len(wp)+1 and gives bin edges, here we make them the same size
            rp = (rp[:-1]+rp[1:])/2
        elif(abs(np.log10(r[2]/r[1]) -np.log10(r[3]/r[2])) <=btol):
            rp = 10**(np.log10(rp[:-1]*rp[1:])/2)
        else:
            raise ValueError("Radial bins are not linear or log spaced. spacing is dr1={0:.2f}, dr2={1:.2f}".format(r[2]-r[1],r[2]-r[1]))

        if(wantGrad):
            return rp,wp,wp_grad
        else:
            return rp,wp


class CrossCorrelator(hzpt):

    def __init__(self,params,hzpt):
        #Set maximum Pade expansion order
        self.params = params
        self.nmax = (len(self.params[1:])-1)//2
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
            b1,pparams = self.params[0],self.params[1:]
            if(wantGrad):
                bb,bbgrad = PBB(k,pparams,nmax=self.nmax,wantGrad=True)
                p = b1*(self.hzpt.P_zel(k) + bb)
                b1_grad = p/b1
                return p, np.hstack([np.atleast_2d(b1_grad).T,b1*bbgrad])
            else: return b1*(self.hzpt.P_zel(k) + PBB(k,pparams,nmax=self.nmax))
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
            b1,pparams = self.params[0],self.params[1:]
            if(wantGrad):
                bb,bbgrad = XiBB(r,pparams,nmax=self.nmax,wantGrad=True)
                xi = b1*(self.hzpt.Xi_zel(r) + bb)
                b1_grad = xi/b1
                return xi, np.hstack([np.atleast_2d(b1_grad).T, b1*bbgrad])
            else: return b1*(self.hzpt.Xi_zel(r) + XiBB(r,pparams,nmax=self.nmax))
        return xi


    '''Copying for now - will move projected statistics to another file later'''
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

        #fast check for bin spacing to return appropriate bin center values
        #r[0] may be 0
        btol=1e-2
        if(abs((r[2]-r[1])-(r[3]-r[2]))<=btol):
            #rp is currently of size len(wp)+1 and gives bin edges, here we make them the same size
            rp = (rp[:-1]+rp[1:])/2
        elif(abs(np.log10(r[2]/r[1]) -np.log10(r[3]/r[2])) <=btol):
            rp = 10**(np.log10(rp[:-1]*rp[1:])/2)
        else:
            raise ValueError("Radial bins are not linear or log spaced. spacing is dr1={0:.2f}, dr2={1:.2f}".format(r[2]-r[1],r[2]-r[1]))
        if(wantGrad):
            return rp,wp,wp_grad
        else:
            return rp,wp

    def Delta_Sigma(self,r,DS0, #PM
                    Ds,Dl,zl,rhom_zl, #background/bin-dependent quantities
                    pi_bins=np.linspace(0,100,10,endpoint=False),rpMin=1.,wantGrad=False):

        """Delta Sigma GGL statistic. Lens redshift is assumed to be the CrossCorrelator attribute z.
        TODO: cosmology gradients?
        Again - no redshift distribution function.
        Parameters:
        -----------
        r: array (float)
            3D abscissa values for xi
        DS0: float
            Point mass marginalization parameter for Delta Sigma (see Singh ++ 2018)
        Ds: float
            Comoving distance (flat) to source
        Dl: float
            Comoving distance (flat) to lens
        zl: float
            Lens redshift
        rhom_zl: float
            Mean matter density at lens redshift
        pi_bins: array (float),optional
            Array must be of size that divides evenly into r - projection window, tophat for now
        rpMin: float,optional
            Minimum scale to which model is trusted, below this activate point mass.
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns:
        ----------
        (array (float), array (float), [array (float)])
            projected radius rp, projected correlation function wp, gradient if wanted
        """
        def inv_Sigma_crit(Ds,Dl,zl):
            c=2.99792e5 #km/s
            G=4.30071e-9 #Mpc (km/s)**2 * M_sun^-1
            #Assume Ds>Dl
            #can come back to computing Ds, DL internally  after caching distances in gzpt object, for now require input
            if(Ds>Dl):
                Dfactor = Dl*(Ds-Dl)/Ds #Mpc/h
            else:
                Dfactor = 0.
            pre = (1+zl)*(4*np.pi*G)/c**2 #Mpc *Msun^-1
            return pre*Dfactor # #Mpc^2 Msun^-1 h^-1, what we need to match units of Sigma
        if(wantGrad):
            rp,wpgm,grad_wpgm = self.wp(r,pi_bins=pi_bins,wantGrad=True)
        else:
            rp,wpgm = self.wp(r,pi_bins=pi_bins)
        #Using Sukhdeep 2018 eqn 29
        #I = np.zeros(len(rp))
        print(rp.shape,wpgm.shape)
        cutmask = rp>=(rpMin-.01)
        wpgm_cut = wpgm[cutmask] #we do not want to use scales in the integration less than r0
        wp_s = ius(rp,wpgm,ext=2)
        wpcut_s = ius(rp[cutmask],wpgm_cut,ext=2)
        S0 = rpMin**2 * (DS0 + rhom_zl*wp_s(rpMin)) #compute Sigma_0 from Singh 18 eqn. 28
        if(wantGrad):
            I_grad_hzpt = np.zeros((len(rp),len(self.params)))
            I_grad_PM = np.zeros(len(rp))
        j=0
        #print('rpcut',rp[cutmask])

        rpint = np.logspace(np.log10(rp.min()),np.log10(rp.max()-0.01),100)
        I = np.zeros(len(rpint))
        I1,I2,I3 = np.zeros(len(rpint)),np.zeros(len(rpint)),np.zeros(len(rpint))
        for i,p in enumerate(rpint):
            if(p<=rpMin): #streamline this later
                term1 = 0. #do not perform the integral over wpgm below rpMin
                j+=1
                term2 = 0.
            else:
                #maybe this is bad...replace with interpolators?
                rr = np.linspace(rpMin,p,100)#len(wpgm_cut[:i-j]))
                #print(rr)
                #print("rr",rr)
                ig = rr*rhom_zl*wpcut_s(rr)#wpgm_cut[:i-j]
                term1 = (1./p**2)*np.trapz(ig,x=rr) #integral term
            #term2 = -rhom_zl*wp_s(p)#wpgm[i] #Sigma_gm, Msun Mpc^-2 h
            term3 =  S0 *(1/p**2) #Sigma_0 ~Pm term
            I1[i] = term1
            I2[i] = term2
            I3[i] = term3
            I[i] = term1 + term2 + term3
            if(wantGrad):
                #just constants and sums
                ig_grad = rhom_zl*(grad_wpgm[:i].T*rr).T
                term1_grad = (1./p**2)*np.trapz(ig,x=rr,axis=0)
                term2_grad = -rhom_zl*grad_wpgm[i]
                I_grad_hzpt[i] = term1_grad+term2_grad
                I_grad_PM[i] = 1/p**2
        if(wantGrad):
            grad_DS = np.concatenate([I_grad_hzpt,np.atleast_2d(I_grad_PM).T],axis=1)
            return rp,I*inv_Sigma_crit(Ds,Dl,zl), grad_DS*inv_Sigma_crit(Ds,Dl,zl)
        else:
            return rpint,I#*inv_Sigma_crit(Ds,Dl,zl),I1*inv_Sigma_crit(Ds,Dl,zl),I2*inv_Sigma_crit(Ds,Dl,zl),I3*inv_Sigma_crit(Ds,Dl,zl),\
            #inv_Sigma_crit(Ds,Dl,zl)#rp,I*inv_Sigma_crit(Ds,Dl,zl)
