'''Real-space tracer/galaxy/halo model'''
from gzpt import hzpt
from gzpt.bb import *
import numpy as np
from numpy import inf
from scipy.special import erf,gamma,poch
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import cumtrapz
from gzpt.utils import W_TH

class AutoCorrelator(hzpt):
    def __init__(self,params,hzpt,params_exc=None,params_sat=None):
        '''
        Parameters
        ----------
        params : list (float)
            List of current hzpt parameter values.
        hzpt : hzpt
            Hzpt base class that holds the linear, zeldovich, and z information.
        params_exc : list,optional
            Exclusion parameters, if desired.
        params_sat : list,optional
            Satellite parameters, if desired.
        '''
        self.params = params
        self.nmax = min((len(self.params[2:])-1)//2,2)
        self.useExc,self.useSat = False,False
        if(params_exc is not None):
            self.useExc = True
            for p in params_exc: self.params.append(p)
        if(params_sat is not None):
            self.useSat = True
            for p in params_sat: self.params.append(p)
        self.hzpt = hzpt


    def Fk_excl(self,k,R_excl,wantGrad=False):
        '''
        Model for the exclusion step.
        Use the Exp model.
        Parameters
        ----------
        k : array (float)
            pts to evaluate P
        R_excl : float
            Exclusion radius
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns
        ---------
        array or (array,array)
            Exclusion kernel, and gradient, if asked for
        '''

        def _hyp0f2(b1,b2, z, eps=1e-6, nmax=10):
            sum = 0
            #accumulate the sum from scratch, no convenient identities, but 5 terms seems good enough, use 10 to be safe
            #mpmath does this but don't want to introduce dependency just for one function
            for k in range(nmax):
                sum+= 1/(poch(b1,k)*poch(b2,k)) * z**k / np.math.factorial(k)
            return sum

        def _fk(k,R_excl):
            #see B.8 of paper
            #try to come back and pass arguments for 0F2 convergence
            t1 = R_excl**3 / 3 * gamma(7/4) * _hyp0f2(1/2,5/4,(k*R_excl/4)**4)
            t2 = -k**2 * R_excl**5 /24 * gamma(5/4) * _hyp0f2(3/2,7/4,(k*R_excl/4)**4)
            return 4*np.pi*(t1 + t2)

        F = -2*_fk(k,R_excl) + _fk(k,2**(-1/4) * R_excl)
        #with these settings can't trust k>10/R h\Mpc
        #Obviously this is not ideal - should use \xi exclusion!
        if(np.any(k>10/R_excl)): F[k>10/R_excl] = 0
        if(wantGrad):
            #analytic gradient of the above expression
            t1 = -R_excl**2 * gamma(7/4) * _hyp0f2(1/2,5/4,k*R_excl/4)
            t2 = -1/210 * k**4 * R_excl**6 * gamma(7/4) * _hyp0f2(3/2,9/4,k*R_excl/4)
            t3 = 1/6 * k**2 * R_excl**4 * gamma(9/4) * _hyp0f2(3/2,7/4,k*R_excl/4)
            t4 = 1/5040 * gamma(9/4) * k**6 * R_excl**8 * _hyp0f2(5/2,11/4,k*R_excl/4)
            grad_R = t1 + t2 + t3 + t4
            if(np.any(k>10)): grad_R[k>10] = 0
            grad_F = [grad_R]
            return F,grad_F
        else:
            return F


    def convolve(self,k,x,y):
        #FIXME most naive implementation, could do better with just about anything else for the integral
        #Do the N^2 operation, assume (probably bad) that not zero padding is fine
        #It would be better (for all N>100) to do the fft convolution
        A,B = ius(k,x,ext=1),ius(k,y,ext=1)
        N = len(k)
        res = np.zeros(N)
        for i in range(N):
            res[i] = np.trapz(k**2 * A(abs(k-k[i]))*B(k),x=k)/(2*np.pi**2)
        return res
        #qualitatively seems to do okay, but should really come back and fix it

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
        #Do some parameter juggling - unfortunate that this is repeated for Xi, FIXME
        if(self.useExc and self.useSat):
            nbar,b1,pparams,eparams = self.params[0],self.params[1],self.params[2:2+(2*self.nmax)+1],self.params[2+(2*self.nmax)+1:-2]
        elif(self.useExc and ~self.useSat):
            nbar,b1,pparams,eparams = self.params[0],self.params[1],self.params[2:2+(2*self.nmax)+1],self.params[2+(2*self.nmax)+1:]
        elif(~self.useExc and self.useSat):
            nbar,b1,pparams = self.params[0],self.params[1],self.params[2:2+(2*self.nmax)+1]
        else:
            nbar,b1,pparams = self.params[0],self.params[1],self.params[2:]

        def Pk(k):
            #Set BB
            if(wantGrad):
                grads=[]
                bb,bbgrad = PBB(k,pparams,nmax=self.nmax,wantGrad=True)
                p = 1/nbar + b1**2 * (self.hzpt.P_zel(k) + bb)
                nbar_grad = -(1/nbar**2) *np.ones(len(k))
                b1_grad = 2*(p- 1/nbar)/b1
                grads.append(np.atleast_2d(nbar_grad).T)
                grads.append(np.atleast_2d(b1_grad).T)
                grads.append(b1**2 * bbgrad)
            else:
                bb = PBB(k,pparams,nmax=self.nmax)
                p = 1/nbar + b1**2 * (self.hzpt.P_zel(k) + bb)

            #Set exclusion
            if(self.useExc):
                if(len(eparams)>1):
                    raise NotImplementedError("Power exclusion is only implemented for the Exp model. Please use FFTLog.")
                R=eparams[0]
                pc = p-1/nbar
                if(wantGrad):
                    exclusion_func,e_f_grad = self.Fk_excl(k,R,wantGrad=True)
                    grad_Re = e_f_grad + self.convolve(k,e_f_grad,pc)
                    grads.append(np.atleast_2d(grad_Re).T)
                else:
                    exclusion_func =  self.Fk_excl(k,R,wantGrad=False) #- (4*np.pi/3)*R**3 * W_TH(k,R)#
                exclusion = exclusion_func + self.convolve(k,exclusion_func,pc)
            else:
                exclusion =0  #no exclusion (additive)

            #put it together
            if(wantGrad): return p + exclusion, np.hstack(grads)
            else: return p + exclusion
        return Pk


    def Fr_excl(self,r,R_excl,sigma_excl=None,wantGrad=False):
        '''
        Model for the exclusion step.
        If no sigma_excl is provided, use the Exp model, o.w. use model of Baldauf++2013.
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
        #Exp model from Appendix B.1
        if(sigma_excl==None):
            f = 1-np.exp(-(r/R_excl)**4)
            F = f**2
            if(wantGrad):
                grad_Re = -8*(r / R_excl)**4 /R_excl * np.exp(-(r/R_excl)**4) * f
                grad_F = [grad_Re,None]
                return F,grad_F
            else:
                return F
        #ExpLog
        #Error function for the exclusion step as in Baldauf++ 2013 eqns (C2), (C4), with log10 as in main text
        else:
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
        Parameters
        ----------
        wantGrad : boolean,optional
            Whether to return the function value, or the tuple (val,grad).
        Returns
        ---------
        callable
            2pcf function of r (Mpc/h)
        '''
        if(self.useExc and self.useSat):
            b1,pparams,eparams,sparams = self.params[1],self.params[2:2+(2*self.nmax)+1],self.params[2+(2*self.nmax)+1:-2],self.params[-2:]
        elif(self.useExc and ~self.useSat):
            b1,pparams,eparams = self.params[1],self.params[2:2+(2*self.nmax)+1],self.params[2+(2*self.nmax)+1:]
            sparams=None
        elif(~self.useExc and self.useSat):
            b1,pparams,sparams = self.params[1],self.params[2:2+(2*self.nmax)+1],self.params[2+(2*self.nmax)+1:]
            eparams=None
        else:
            b1,pparams,eparams = self.params[1],self.params[2:],None
            eparams,sparams=None,None

        def xi(r):
            #Set BB
            if(wantGrad):
                grads=[]
                bb,bbgrad = XiBB(r,pparams,nmax=self.nmax,wantGrad=True)
                xic = b1**2 * (self.hzpt.Xi_zel(r) + bb)
                b1_grad = 2.*xic/b1
                grads.append(np.atleast_2d(b1_grad).T)
                grads.append(b1**2 * bbgrad)
            else:
                bb = XiBB(r,pparams,nmax=self.nmax)
                xic = b1**2 * (self.hzpt.Xi_zel(r) + bb)


            #Set exclusion
            if(self.useExc):
                if(len(eparams)==2): #check for Exp or ExpLog model
                    R,sigma = eparams
                else:
                    R,sigma = eparams,None
                if(wantGrad):
                    exclusion,e_grad = self.Fr_excl(r,R,sigma,wantGrad=True)
                    grad_Re,grad_sigmae = e_grad
                    grads.append(np.atleast_2d(grad_Re*(1. + xic)).T)
                    grads.append(np.atleast_2d(grad_sigmae*(1. + xic)).T)
                else:
                    print('R,sigma', R,sigma)
                    exclusion = self.Fr_excl(r,R,sigma)
            else:
                exclusion=1 #no exclusion (multiplicative)

            #Set sattelites
            if(self.useSat):
                R_s_fixed=1e3 #fix compensation to effectively infinity
                A1s,R1h1s = sparams
                if(wantGrad):
                    satellite, satellite_grad = XiBB(r,[A1s,R_s_fixed,R1h1s],nmax=1,wantGrad=True)
                    grads.append(b1**2 * np.array([satellite_grad[:,0],satellite_grad[:,1]]).T)
                else:
                    satellite = XiBB(r,[A1s,R_s_fixed,R1h1s],nmax=1)
            else:
                satellite=0 #no sats (additive)

            #put it together
            xi_d = exclusion*(1. + xic) - 1.
            if(wantGrad): return xi_d + satellite, np.hstack(grads)
            else: return xi_d + satellite
        return xi

    def wp(self,rp,pi=np.linspace(0,100,100+1),rMin=0.01,wantGrad=False):
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

        if(wantGrad):
            wp_grad=np.zeros((len(wp),len(self.params)))
            xir,grad_xir = self.Xi(wantGrad=wantGrad)
        else:
            xir = self.Xi(wantGrad=wantGrad)
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

        if(wantGrad):
            wp_grad=np.zeros((len(rp),len(self.params)))
            xir,grad_xir = self.Xi(wantGrad=wantGrad)
        else:
            xir = self.Xi(wantGrad=wantGrad)

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
        Simple top-hat redshift distribution.
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
        #FIXME: do not actually need this (it works though)- we are not computing tangential shear...might later
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

        #FIXME: internally check units on this, works in the example though

        if(wantGrad):
            wpgm,grad_wpgm = self.wp(rp,pi=pi,wantGrad=True)

        else:
            wpgm = self.wp(rp,pi=pi)
        #rp interpolation pts
        rpint=np.logspace(np.log10(rp.min()),np.log10(rp.max()),num_rpint) #FIXME probably not necessary to use 1000

        if(wantGrad):
            I_grad_hzpt = np.zeros((len(rpint),len(self.params)))
            I_grad_PM = np.zeros(len(rpint))

        wint = np.interp(rpint,rp,wpgm)
        rpint_c = rpint[rpint>=rpMin]
        wint_c = wint[rpint>=rpMin]
        s_int = rhom*np.interp(rpint,rp,wpgm)
        s = rhom*wpgm
        sint_c = rhom*wint_c
        t1=np.zeros(len(rpint))
        t1[rpint<rpMin] = 0.
        #Using Singh 2018 eqn 29 - there is a typo so add missing factor of 2
        t1[rpint>=rpMin] = (2./rpint_c**2)*cumtrapz(rpint_c*sint_c,x=rpint_c,initial=0)
        t2 = -s_int
        S0=np.interp(rpMin,rpint,s_int)
        t3 = (rpMin/rpint)**2 * (DS0 + S0) #compute Sigma_0 from Singh 18 eqn. 30
        DS=t1+t2+t3

        if(wantGrad): #FIXME: test for this
            grad_wpgm_int_c = np.zeros((len(rpint_c),grad_wpgm.shape[1]))
            for i in range(grad_wpgm.shape[1]):
                grad_wpgm_int[:,i] = np.interp(rpint,rp,grad_wpgm[:,i])
                grad_wpgm_int_c[:,i] = grad_wpgm_int[:,i][rpint>=rpMin]
                term1_grad[:,i] = (2./rpint**2)*cumtrapz(rpint_c*rhom*grad_wpgm_int_c,x=rpint_c,axis=0,initial=0)
            term2_grad = -rhom*grad_wpgm
            I_grad_hzpt = term1_grad+term2_grad
            I_grad_PM = 1/rpint**2

        if(wantGrad):
            grad_DS = np.concatenate([I_grad_hzpt,np.atleast_2d(I_grad_PM).T],axis=1)
            return DS,grad_DS
        else:
            return DS
