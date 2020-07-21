'''Real-space tracer/galaxy/halo model'''
from gzpt import hzpt
import zel,bb
import numpy as np
from numpy import inf
from scipy.special import erf
from utils import rDelta

class AutoCorrelator(hzpt):
    '''hzpt object provides:
    '''
    def __init__(self,hzpt,nmax,useExc=True):
        #Set maximum Pade expansion order
        self.nmax = nmax
        self.z = hzpt.z#redshift
        self.loud=False

        #making this up for right now - using lowZ nbar,bias made up will eventually put some fit function e.g. tinker
        self.useExc = useExc
        nbar_init,b1_init=4.2e-4, 2.
        A0_init,R_init =
        R1h_init =
        R1sq_init,R2h_init =
        Rexcl_init,sigmaexcl_init =
        A0_1h_init,R1h_1h_init =

        if(self.nmax==0):
            self.params = np.array([nbar_init,b1_init,A0_init,R_init]) #A0, R
        elif(self.nmax==1):
            if(self.useExc):
                self.params = np.array([nbar_init,b1_init,A0_init,R_init,R1h_init,Rexcl_init,sigmaexcl_init]) #A0, R, R1h
            else:
                self.params = np.array([nbar_init,b1_init,A0_init,R_init,R1h_init]) #A0, R, R1h
        elif(self.nmax==2):
            if(self.useExc):
                self.params = np.array([nbar_init,b1_init,A0_init,R_init,R1h_init,Rexcl_init,sigmaexcl_init,A0_1h_init,R1h_1h_init])
            else:
                self.params = np.array([nbar_init,b1_init,A0_init,R_init,R1h_init,R1sq_init,R2h_init]) #A0, R, R1h, R1, R2h I think
        else:
            print('nmax not supported')
            '''TODO: Throw error'''

    def Power(self,params=[]):
        '''Returns callable that incorporates the current parameters'''
        if(not self.useExc):
            if(len(params)<1): nbar,b1,pparams = self.params[0],self.params[1],self.params[2:]
            else: nbar,b1,pparams = params[0],params[1],params[2:]
        else:
            if(len(params)<1): nbar,b1,pparams = self.params[0],self.params[1],self.params[2:-2],self.params[-2:]
            else: nbar,b1,pparams = params[0],params[1],params[2:-2],params[-2:]
        def Pk(k):
            '''add interpolator in z also? also don't forget Xi'''
            return 1/nbar + b1**2 * (self.P_zel(k,self.z) + bb.PBB(k,pparams,nmax=self.nmax))
            #In theory the above in parentheses should be exactly the same as Pmm, though b/c of sattelites/subhalos, PBB will change SD
        return Pk

    # def Power_Baldauf(self,params=[]):
    #     '''Returns callable that incorporates the current parameters'''
    #     if(len(params)<1): nbar,b1,pparams = self.params[0],self.params[1],self.params[2:]
    #     else: nbar,b1,pparams = params[0],params[1],params[2:]
    #     def Pk(k):
    #         '''add interpolator in z also? also don't forget Xi'''
    #         return 1/nbar + b1**2 * (self.P_zel(k,self.z) + bb.PBB(k,pparams,nmax=self.nmax))
    #         #In theory the above in parentheses should be exactly the same as Pmm, though b/c of sattelites/subhalos, PBB will change SD
    #     return Pk

    '''FIXME: This repetition of this function for different nmax is lazy and unneccesary'''

    def grad_Power0(self,k,nbar,b1,A0,R):
        params = np.array([nbar,b1,A0,R])
        def gradP(k):
            nbar_grad = -(1/nbar**2) *np.ones(len(k))
            '''Fitting bias independently from BB parameters'''
            b1_grad = 2*(self.Power(params)(k)- 1/nbar)/b1
            return np.hstack([np.atleast_2d(nbar_grad).T,np.atleast_2d(b1_grad).T,
                              b1**2 * bb.PBB(k,params[2:],nmax=self.nmax,wantGrad=True)[1]])
        return gradP(k)

    def grad_Power1(self,k,nbar,b1,A0,R,R1h):
        params = np.array([nbar,b1,A0,R,R1h])
        def gradP(k):
            nbar_grad = -(1/nbar**2) *np.ones(len(k))
            '''Fitting bias independently from BB parameters'''
            b1_grad = 2*(self.Power(params)(k)- 1/nbar)/b1
            #bb gradient is len(k) X npparams so want nbar and b1 to be columns of len(k) i.e. len(k) x 1
            return np.hstack([np.atleast_2d(nbar_grad).T,np.atleast_2d(b1_grad).T,
                              b1**2 * bb.PBB(k,params[2:],nmax=self.nmax,wantGrad=True)[1]])
        return gradP(k)

    def grad_Power2(self,k,nbar,b1,A0,R,R1h,R1sq,R2h):
        params = np.array([nbar,b1,A0,R,R1h,R1sq,R2h])
        def gradP(k):
            nbar_grad = -(1/nbar**2) *np.ones(len(k))
            '''Fitting bias independently from BB parameters'''
            b1_grad = 2*(self.Power(params)(k)- 1/nbar)/b1
            return np.hstack([np.atleast_2d(nbar_grad).T,np.atleast_2d(b1_grad).T,
                              b1**2 * bb.PBB(k,params[2:],nmax=self.nmax,wantGrad=True)[1]])
        return gradP(k)

    def F_excl(self,r,R_excl,sigma_excl=0.09):
        return .5*(1 + erf(np.log10(r/R_excl)/(np.sqrt(2)*sigma_excl)))  #Baldauf++ 2013 eqns (C2), (C4), with log10 as in main text

    def Xi(self,params=[]):#,R_excl=None,sigma_excl=0.09):
        '''Returns callable that incorporates the current parameters'''
        if(self.useExc):
            if(self.nmax==1):
                if(len(params)<1): b1,pparams,eparams = self.params[1],self.params[2:-2],self.params[-2:]
                else: b1,pparams,eparams= params[0],params[1:-2],params[-2:]
            elif(self.nmax==2):
                if(len(params)<1): b1,pparams,eparams,ohparams = self.params[1],self.params[2:-4],self.params[-4:-2],self.params[-2:]
                else: b1,pparams,eparams,ohparams = params[0],params[1:-4],params[-4:-2],params[-2:]
            R_excl,sigma_excl = eparams[0],eparams[1]
        else:
            if(len(params)<1): b1,pparams,eparams = self.params[1],self.params[2:],None
            else: b1,pparams,eparams = params[0],params[1:], None

        def xi(r):
            '''maybe put exclusion in separate function'''

            if(self.useExc):
                R,sigma = eparams
                exclusion_baldauf = self.F_excl(r,R,sigma)#.5*(1 + erf(np.log10(r/R)/(np.sqrt(2)*sigma)))
                exclusion = exclusion_baldauf
            else:
                exclusion=1.
            if(self.nmax==2 and self.useExc):
                xihzpt = b1**2 * (self.Xi_zel(r,self.z)+ bb.XiBB(r,pparams,nmax=1))
                xi_baldauf_d = exclusion*(1+xihzpt)-1
                return xi_baldauf_d +  bb.XiBB(r,[ohparams[0],1e3,ohparams[1]],nmax=1) #set compensation R to existing value
            else:
                xihzpt = b1**2 * (self.Xi_zel(r,self.z)+ bb.XiBB(r,pparams,nmax=self.nmax))
                xi_baldauf_d = exclusion*(1+xihzpt)-1
                return xi_baldauf_d
        return xi

    def grad_Xi0(self,r,b1,A0,R):
        params = np.array([b1,A0,R])
        b1_grad = 2*self.Xi(params)(r)/b1
        def gradX(r):
            return np.hstack([np.atleast_2d(b1_grad).T,
                                                    b1**2 * bb.XiBB(r,params[1:],nmax=self.nmax,wantGrad=True)[1]])
        return gradX(r)

    def grad_Xi1(self,r,b1,A0,R,R1h,Re,sigmae):
        params = np.array([b1,A0,R,R1h,Re,sigmae])
        pparams = params[1:-2]
        self.useExc=False
        b1_grad = 2*self.Xi(params[:-2])(r)/b1
        self.useExc=True
        #b1_grad = 2.*((self.Xi(params)(r)+1)/self.F_excl(r,Re,sigmae) -1)/b1
        #print(b1_grad)
        def gradX(r):
            ln10 = np.log(10)
            exp = -np.exp(-np.log(r/Re)**2 / (2*sigmae**2 * ln10**2))
            denom = np.sqrt(2.*np.pi)* sigmae * ln10
            grad_Re = exp*denom/Re
            grad_sigmae = exp*denom*np.log(r/Re)/sigmae
            indep = b1_grad*b1/2. +1.  #the part that doesn't depend on exclusion parameters
            return np.hstack([np.atleast_2d(b1_grad).T,
                                                    b1**2 * bb.XiBB(r,pparams,nmax=self.nmax,wantGrad=True)[1],
                                                    np.atleast_2d(grad_Re*indep).T,np.atleast_2d(grad_sigmae*indep).T])
        return gradX(r)

    def grad_Xi1_no_exc(self,r,b1,A0,R,R1h):
        params = np.array([b1,A0,R,R1h])
        pparams = params[1:4]
        b1_grad = 2*self.Xi(params)(r)/b1
        def gradX(r):
            return np.hstack([np.atleast_2d(b1_grad).T,
                                                    b1**2 * bb.XiBB(r,pparams,nmax=self.nmax,wantGrad=True)[1]])
        return gradX(r)

    def grad_Xi2(self,r,b1,A0,R,R1h,Re,sigmae,A0_1h,R1h_1h):
        params = np.array([b1,A0,R,R1h,Re,sigmae,A0_1h,R1h_1h])
        pparams = params[1:-4]
        eparams = params[-4:-2]
        ohparams = params[-2:]
        self.useExc=False
        b1_grad = 2*self.Xi(params[:-4])(r)/b1
        self.useExc=True

        def gradX(r):
            ln10 = np.log(10)
            exp = -np.exp(-np.log(r/Re)**2 / (2*sigmae**2 * ln10**2))
            '''There is a missing factor of 1/x here!'''
            denom = np.sqrt(2.*np.pi)* sigmae * ln10
            grad_Re = exp*denom/Re
            grad_sigmae = exp*denom*np.log(r/Re)/sigmae
            indep = b1_grad*b1/2. +1.  #the part that doesn't depend on exclusion parameters
            return np.hstack([np.atleast_2d(b1_grad).T,
                                                    b1**2 * bb.XiBB(r,pparams,nmax=self.nmax,wantGrad=True)[1],
                                                    np.atleast_2d(grad_Re*indep).T,np.atleast_2d(grad_sigmae*indep).T,
                                                    b1**2 * bb.XiBB(r,ohparams,nmax=1,wantGrad=True)[1][0],
                                                    b1**2 * bb.XiBB(r,ohparams,nmax=1,wantGrad=True)[1][2]])
        return gradX(r)

    def grad_Xi2_no_exc(self,r,b1,A0,R,R1h,R1sq,R2h):
        params = np.array([b1,A0,R,R1h,R1sq,R2h])
        b1_grad = 2*self.Xi(params)(r)/b1
        def gradX(r):
            return np.hstack([np.atleast_2d(b1_grad).T,
                                                    b1**2 * bb.XiBB(r,params[1:],nmax=self.nmax,wantGrad=True)[1]])
        return gradX(r)

    def wp(self,r,pi_bins=np.linspace(0,100,10,endpoint=False)):
        """Projected correlation function - no Kaiser or redshift distribution.
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

    def wp_grad():
        """Simply integrate up the existing gradients for xi in the bins in a trapz"""
        return None


class CrossCorrelator(hzpt):

    def __init__(self,hzpt,redshift,nmax):
        #Set maximum Pade expansion order
        self.nmax = nmax
        self.z = redshift
        self.loud=False

        #hacky but alternative is one really long file, which I may move to in the future
        self.cosmo,self.z_pre,self.k_pre,self.r_pre,self.tablePZel,self.tableXiZel,self.sigma8_pre = hzpt.cosmo,hzpt.z_pre,hzpt.k_pre,hzpt.r_pre,hzpt.tablePZel,hzpt.tableXiZel,hzpt.sigma8_pre

        #Power laws in s8 from pyRSD P_mm may want to update these
        b1_init=2.
        A0_init,R_init = 708.*(self.sigma8(redshift)/0.8)**3.65,31.8*(self.sigma8(redshift)/0.8)**0.13
        R1h_init = 3.77*(self.sigma8(redshift)/0.8)**-0.10
        R1sq_init,R2h_init = (3.24*(self.sigma8(redshift)/0.8)**0.37)**2,1.70*(self.sigma8(redshift)/0.8)**0.42


        if(self.nmax==0):
            self.params = np.array([b1_init,A0_init,R_init]) #A0, R
        elif(self.nmax==1):
            self.params = np.array([b1_init,A0_init,R_init,R1h_init ]) #A0, R, R1h
        elif(self.nmax==2):
            self.params = np.array([b1_init,A0_init,R_init,R1h_init, R1sq_init,R2h_init]) #A0, R, R1h, R1, R2h I think
        else:
            print('nmax not supported')
            '''TODO: Throw error'''

        #set bounds from file
        priors_low,priors_high = np.loadtxt('./bounds.ini')[5:len(self.params)+5].T
        self.bounds = [[*priors_low],[*priors_high]]


    def Power(self,params=[]):
        '''Returns callable that incorporates the current parameters'''
        if(len(params)<1): b1,pparams = self.params[0],self.params[1:]
        else: b1,pparams = params[0],params[1:]
        def Pk(k):
            '''add interpolator in z also? also don't forget Xi'''
            return  b1 * (self.P_zel(k,self.z) + bb.PBB(k,pparams,nmax=self.nmax))
        return Pk

    '''FIXME: This repetition of this function for different nmax is lazy and unneccesary'''

    def grad_Power0(self,k,b1,A0,R):
        params = np.array([b1,A0,R])
        def gradP(k):
            '''Fitting bias independently from BB parameters'''
            b1_grad = self.Power(params)(k)/b1
            return np.hstack([np.atleast_2d(b1_grad).T,
                              b1* bb.PBB(k,params[1:],nmax=self.nmax,wantGrad=True)[1]])
        return gradP(k)

    def grad_Power1(self,k,b1,A0,R,R1h):
        params = np.array([b1,A0,R,R1h])
        def gradP(k):
            '''Fitting bias independently from BB parameters'''
            b1_grad = self.Power(params)(k)/b1
            return np.hstack([np.atleast_2d(b1_grad).T,
                              b1* bb.PBB(k,params[1:],nmax=self.nmax,wantGrad=True)[1]])
        return gradP(k)

    def grad_Power2(self,k,b1,A0,R,R1h,R1sq,R2h):
        params = np.array([b1,A0,R,R1h,R1sq,R2h])
        def gradP(k):
            '''Fitting bias independently from BB parameters'''
            b1_grad = self.Power(params)(k)/b1
            return np.hstack([np.atleast_2d(b1_grad).T,
                              b1* bb.PBB(k,params[1:],nmax=self.nmax,wantGrad=True)[1]])
        return gradP(k)

    def Xi(self,params=[]):
        '''Returns callable that incorporates the current parameters'''
        if(len(params)<1): b1,pparams = self.params[0],self.params[1:]
        else: b1,pparams = params[0],params[1:]
        def xi(r):
            return b1 * (self.Xi_zel(r,self.z)+ bb.XiBB(r,pparams,nmax=self.nmax))
            #return r * b1 * (self.Xi_zel(r,self.z)+ bb.XiBB(r,pparams,nmax=self.nmax))
            #return r**2 * b1 * (self.Xi_zel(r,self.z)+ bb.XiBB(r,pparams,nmax=self.nmax))
        return xi

    def grad_Xi0(self,r,b1,A0,R):
        params = np.array([b1,A0,R])
        b1_grad = self.Xi(params)(r)/b1
        def gradX(r):
            return np.hstack([np.atleast_2d(b1_grad).T,
                                                    b1**2 * bb.XiBB(r,params[1:],nmax=self.nmax,wantGrad=True)[1]])

        return gradX(r)

    def grad_Xi1(self,r,b1,A0,R,R1h):
        params = np.array([b1,A0,R,R1h])
        b1_grad = self.Xi(params)(r)/b1
        def gradX(r):
            return np.hstack([np.atleast_2d(b1_grad).T,
                                                    b1**2 * bb.XiBB(r,params[1:],nmax=self.nmax,wantGrad=True)[1]])

        return gradX(r)

    def grad_Xi2(self,r,b1,A0,R,R1h,R1sq,R2h):
        params = np.array([b1,A0,R,R1h,R1sq,R2h])
        b1_grad = self.Xi(params)(r)/b1
        def gradX(r):
            return np.hstack([np.atleast_2d(b1_grad).T,
                                                    b1**2 * bb.XiBB(r,params[1:],nmax=self.nmax,wantGrad=True)[1]])

        return gradX(r)



    def wp(self,r,pi_bins=np.linspace(0,100,10,endpoint=False)):
        "FIXME: Lazy copy from AutoCorrelator - should make this function accessible by both. - general correlator class..."
        """Projected correlation function - no Kaiser or redshift distribution.
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
        #as for autocorrelator
        return None

    def Delta_Sigma(self,r,z_source,S0=12,pi_bins=np.linspace(0,100,10,endpoint=False),rpMin=0.1):
        """Delta Sigma GGL statistic. Lens redshift is assumed to be the CrossCorrelator attribute z.
        Again - no redshift distribution function."""
        def inv_Sigma_crit(zs,zl):
            c=3e5 #km/s
            G=4.3e-9 #Mpc (km/s)**2 * M_sun
            #Assume Ds>Dl
            Ds,Dl = self.cosmo.comoving_transverse_distance(zs),self.cosmo.comoving_transverse_distance(zl)
            if(Ds>Dl):
                Dfactor = Dl*(Ds-Dl)/(Ds)
            else:
                Dfactor = 0.
            pre = (1+zl)*(4*np.pi*G)/c**2
            return pre*Dfactor
        z_lens = self.z
        rho = (self.cosmo.rho_m(z_lens)*1e10)
        rp,wpgm = self.wp(r,pi_bins=pi_bins) #should be okay?
        #Using Sukhdeep 2018 eqn 29
        I = np.zeros(len(rp))
        for i,p in enumerate(rp):
            rr = np.linspace(rpMin,p,len(wpgm[:i]))
            ig = rr*rho*wpgm[:i]
            term1 = (1./p**2)*np.trapz(ig,x=rr) #integral term
            term2 = -rho*wpgm[i] #Sigma_gm
            term3 =  S0 *(1/p**2) #Sigma_0 ~Pm term
            I[i] = term1 + term2 + term3
        return rp,I*inv_Sigma_crit(z_source,z_lens)

    def grad_Delta_Sigma():
        "I think this should be similar to the wp grad - just sum things up or whatever for each DS term
        term 1 will be a sum over wp, double grad some like for wp. term 2 just wp, term 3 PM
        TODO the complication of gradient wrt cosmology for rho"
        return None
