'''Real-space tracer/galaxy/halo model'''
from gzpt import hzpt
from gzpt.bb import *
import numpy as np
from numpy import inf
from scipy.special import erf
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import cumtrapz

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
                return 1/nbar + b1**2 * (self.hzpt.P_zel(k) + bb)
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
    def wp(self,rp,pi=np.linspace(0,100,100+1),rMin=0.01,wantGrad=False):
        "FIXME: Lazy copy from AutoCorrelator - should make this function accessible by both. - general correlator class..."
        """Projected correlation function.
        Parameters:
        -----------
        rp: array (float)
            2D abscissa values for wp
        pi_bins: array (float)
            Array must be of size that divides evenly into r - projection window, tophat for now
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns:
        ----------
        (array (float), array (float), [array (float)])
            projected radius rp, projected correlation function wp, gradient if wanted
        """

        # dpi = pi[1]-pi[0] #bin width
        # #pi_e = pi+dpi/2 #bin edges
        # pi_pts = np.concatenate([-pi[::-1],pi[1:]]) #add negative points, now don't need factor of 2 in wp
        # #I think this might not strictly be what follows the text of Singh++16 but for the low-rp values need a point at pi=0
        # #O.w. will underestimate wp below r=5-10, another solution to this would be log binning pi but this is not done bc real bins I guess are linear
        #
        if(wantGrad):
            wp_grad=np.zeros((len(wp),len(self.params)))
            xir,grad_xir = self.Xi(wantGrad=wantGrad)
        else:
            xir = self.Xi(wantGrad=wantGrad)
        # #given now rp = np.logspace(np.log10(r.min()),np.log10(r.max()),100)
        # wp = np.zeros(len(rp))
        # for i in range(len(wp)):
        #     rev=np.sqrt(rp[i]**2 + pi_pts**2)
        #     wp[i] = dpi * np.sum(xir(rev))
        #     if(wantGrad): wp_grad[i] = dpi * np.sum(grad_xir(rev),axis=0) #not sure if this works
        #
        #wp - yet again
        dpi = pi[1]-pi[0] #bin width
        wp = np.zeros(len(rp))
        for i in range(len(wp)):
            rev=np.sqrt(rp[i]**2 + pi**2)
            wp[i]= 2*ius(rev,xir(rev)/np.sqrt(1- (rp[i]/(rev+rMin))**2)).integral(rev.min(),rev.max())
            if(wantGrad): wp_grad[i] =2*ius(rev,grad_xir(rev)/np.sqrt(1- (rp[i]/(rev+rMin))**2)).integral(rev.min(),rev.max())


        if(wantGrad):
            return wp,wp_grad
        else:
            return wp



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
    def wp(self,rp,pi=np.linspace(0,100,100+1),rMin=0.1,wantGrad=False):
        "FIXME: Lazy copy from AutoCorrelator - should make this function accessible by both. - general correlator class..."
        """Projected correlation function.
        Parameters:
        -----------
        rp: array (float)
            2D abscissa values for wp
        pi_bins: array (float)
            Array must be of size that divides evenly into r - projection window, tophat for now
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns:
        ----------
        (array (float), array (float), [array (float)])
            projected radius rp, projected correlation function wp, gradient if wanted
        """

        #dpi = pi[1]-pi[0] #bin width
        #pi_e = pi+dpi/2 #bin edges
        #pi_pts = np.concatenate([-pi[::-1],pi[1:]]) #add negative points, now don't need factor of 2 in wp
        #I think this might not strictly be what follows the text of Singh++16 but for the low-rp values need a point at pi=0
        #O.w. will underestimate wp below r=5-10, another solution to this would be log binning pi but this is not done bc real bins I guess are linear

        if(wantGrad):
            wp_grad=np.zeros((len(rp),len(self.params)))
            xir,grad_xir = self.Xi(wantGrad=wantGrad)
        else:
            xir = self.Xi(wantGrad=wantGrad)
        # #given now rp = np.logspace(np.log10(r.min()),np.log10(r.max()),100)
        # wp = np.zeros(len(rp))
        # for i in range(len(wp)):
        #     rev=np.sqrt(rp[i]**2 + pi_pts**2)
        #     wp[i] = dpi * np.sum(xir(rev))
        #     if(wantGrad): wp_grad[i] = dpi * np.sum(grad_xir(rev),axis=0) #not sure if this works
        #
        # if(wantGrad):
        #     return wp,wp_grad
        # else:
        #     return wp

        #wp - yet again
        dpi = pi[1]-pi[0] #bin width
        wp = np.zeros(len(rp))
        for i in range(len(wp)):
            rev=np.sqrt(rp[i]**2 + pi**2)
            wp[i]= 2*ius(rev,xir(rev)/np.sqrt(1- (rp[i]/(rev+rMin))**2)).integral(rev.min(),rev.max())
            if(wantGrad): wp_grad[i] =2*ius(rev,grad_xir(rev)/np.sqrt(1- (rp[i]/(rev+rMin))**2)).integral(rev.min(),rev.max())


        if(wantGrad):
            return wp,wp_grad
        else:
            return wp

    def Delta_Sigma(self,rp,DS0, #PM
                    Ds,Dl,zl,rhom, #background/bin-dependent quantities
                    pi=np.linspace(0,100,100+1),rpMin=1.,num_rpint=1000,wantGrad=False):

        """Delta Sigma GGL statistic. Lens redshift is assumed to be the CrossCorrelator attribute z.
        TODO: cosmology gradients?
        Again - no redshift distribution function.
        Parameters:
        -----------
        rp: array (float)
            2D abscissa values for wp/DS
        DS0: float
            Point mass marginalization parameter for Delta Sigma (see Singh ++ 2018)
        Ds: float
            Comoving distance (flat) to source
        Dl: float
            Comoving distance (flat) to lens
        zl: float
            Lens redshift
        rhom: float
            Mean matter density at z=0, in Msun^1 pc^-2 h^1 (pc, NOT Mpc)
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
        #FIXME: turns out do not actually need this (it works though)- we are not computing tangential shear...might later
        # def inv_Sigma_crit(Ds,Dl,zl):
        #     c=2.99792e5 #km/s
        #     G=4.30071e-9 #Mpc (km/s)**2 * M_sun^-1
        #     #Assume Ds>Dl
        #     #can come back to computing Ds, DL internally  after caching distances in gzpt object, for now require input
        #     if(Ds>Dl):
        #         Dfactor = Dl*(Ds-Dl)/Ds #Mpc/h
        #     else:
        #         Dfactor = 0.
        #     pre = (1+zl)*(4*np.pi*G)/c**2 #Mpc *Msun^-1
        #     return pre*Dfactor # #Mpc^2 Msun^-1 h^-1, what we need to match units of Sigma


        if(wantGrad):
            #rp,wpgm,grad_wpgm = self.wp(r,pi_bins=pi_bins,wantGrad=True)
            wpgm,grad_wpgm = self.wp(rp,pi=pi,wantGrad=True)

        else:
            #rp,wpgm = self.wp(r,pi_bins=pi_bins)
            wpgm = self.wp(rp,pi=pi)
            #test where we know the answer
            #rp,_,_,wpgm,_,_,_,_,_,_,_,_ = np.loadtxt('/Users/jsull/tmp_cori_maintenence/mocks_sukhdeep/evol_mock/evol_DM1_r00.2_w.dat',unpack=True)
        #Using Sukhdeep 2018 eqn 29 - there is a typo so add missing factor of 2
        #I = np.zeros(len(rp))
        # cutmask = rp>=(rpMin-.01)
        # wpgm_cut = wpgm[cutmask] #we do not want to use scales in the integration less than r0
        # wp_s = ius(rp,wpgm,ext=2)
        # wpcut_s = ius(rp[cutmask],wpgm_cut,ext=2)
        # S0 = rpMin**2 * (DS0 + rhom_zl*wp_s(rpMin))
        #rp interpolation pts
        rpint=np.logspace(np.log10(rp.min()),np.log10(rp.max()),num_rpint) #FIXME probably not necessary to use 1000

        if(wantGrad):
            # I_grad_hzpt = np.zeros((len(rp),len(self.params)))
            # I_grad_PM = np.zeros(len(rp))
            I_grad_hzpt = np.zeros((len(rpint),len(self.params)))
            I_grad_PM = np.zeros(len(rpint))

        # rp_c = rp[rp>=rMin]
        # wpgm_c = wpgm[rp>=rMin]
        wint = np.interp(rpint,rp,wpgm)
        rpint_c = rpint[rpint>=rpMin]
        wint_c = wint[rpint>=rpMin]
        s_int = rhom*np.interp(rpint,rp,wpgm)
        s = rhom*wpgm
        #s_c = rhom*wpgm_c
        sint_c = rhom*wint_c
        t1=np.zeros(len(rpint))
        t1[rpint<rpMin] = 0.
        t1[rpint>=rpMin] = (2./rpint_c**2)*cumtrapz(rpint_c*sint_c,x=rpint_c,initial=0)
        t2 = -s_int
        S0=np.interp(rpMin,rpint,s_int)
        t3 = (rpMin/rpint)**2 * (DS0 + S0) #compute Sigma_0 from Singh 18 eqn. 30
        DS=t1+t2+t3

        if(wantGrad): #This probably wont work so come back and check it
            #just constants and sums
            grad_wpgm_int_c = np.zeros((len(rpint_c),grad_wpgm.shape[1]))
            for i in range(grad_wpgm.shape[1]):
                grad_wpgm_int[:,i] = np.interp(rpint,rp,grad_wpgm[:,i])
                grad_wpgm_int_c[:,i] = grad_wpgm_int[:,i][rpint>=rpMin]
                term1_grad[:,i] = (2./rpint**2)*cumtrapz(rpint_c*rhom*grad_wpgm_int_c,x=rpint_c,axis=0,initial=0)
            term2_grad = -rhom*grad_wpgm
            I_grad_hzpt = term1_grad+term2_grad
            I_grad_PM = 1/rpint**2
        # j=0
        # rpint = np.logspace(np.log10(rp.min()),np.log10(rp.max()-0.01),100)
        # I = np.zeros(len(rpint))
        # I1,I2,I3 = np.zeros(len(rpint)),np.zeros(len(rpint)),np.zeros(len(rpint))
        #
        #
        #
        # for i,p in enumerate(rpint):
        #     if(p<=rpMin): #streamline this later
        #         term1 = 0. #do not perform the integral over wpgm below rpMin
        #         j+=1
        #     else:
        #         #maybe this is bad...replace with interpolators?
        #         rr = np.linspace(rpMin,p,100)#len(wpgm_cut[:i-j]))
        #         ig = rr*rhom_zl*wpcut_s(rr)#wpgm_cut[:i-j]
        #         term1 = (2./p**2)*np.trapz(ig,x=rr) #integral term
        #     term2 = -rhom_zl*wp_s(p)#wpgm[i] #Sigma_gm, Msun Mpc^-2 h
        #     term3 =  S0 *(1/p**2) #Sigma_0 ~Pm term
        #     I1[i] = term1
        #     I2[i] = term2
        #     I3[i] = term3
        #     I[i] = term1 + term2 + term3
        #     if(wantGrad):
        #         #just constants and sums
        #         ig_grad = rhom_zl*(grad_wpgm[:i].T*rr).T
        #         term1_grad = (2./p**2)*np.trapz(ig,x=rr,axis=0)
        #         term2_grad = -rhom_zl*grad_wpgm[i]
        #         I_grad_hzpt[i] = term1_grad+term2_grad
        #         I_grad_PM[i] = 1/p**2


        if(wantGrad):
            grad_DS = np.concatenate([I_grad_hzpt,np.atleast_2d(I_grad_PM).T],axis=1)
            #return rpint,DS,grad_DS
            return DS,grad_DS
        else:
            #return rpint,DS
            return DS
