"""Compute 2pt stats in zeldovich approximation

"""
#imports
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
import pyfftw
from scipy.special import loggamma
from scipy.interpolate import interp1d
import numpy as np

def loginterp(x, y, yint = None, side = "both", lorder = 9, rorder = 9, lp = 1, rp = -1,
              ldx = 1e-6, rdx = 1e-6):
    '''
    Extrapolate function by evaluating a log-index of left & right side.
    Use 9th order FD derivative to find effective power law index and use that power law for extrapolation.
    From Chirag Modi's CLEFT code at
    https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py
    The warning for divergent power laws on both ends is turned off.

    Parameters:
    -----------
    x: array
        abscissa values
    y: array
        funtion values
    yint: callable,optional
        interpolator callable
    side: string, optional:
        one of 'l','r',or 'both' - edge to extrapolate
    lorder,rorder: int, optional
        left and right order of FD derivative evalauted at edges
    lp,rp: int,optional
        left and right extrap indices, assumes first and last elements
    lidx,ridx: float,optional
        left and right FD epsilon value

    Returns:
    -----------
    callable
        interpolated univariate spline over the extrapolated range
    '''
    if yint is None:
        yint = interpolate(x, y, k = 5)
    if side == "both":
        side = "lr"

    #get derivatives
    l =lp
    r =rp
    lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
    rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]

    #the exrapolated power laws
    xl = np.logspace(-12, np.log10(x[l]), 10**5)
    xr = np.logspace(np.log10(x[r]), 12., 10**5)
    yl = y[l]*(xl/x[l])**lneff
    yr = y[r]*(xr/x[r])**rneff

    #put it together
    xint = x[l+1:r].copy()
    yint = y[l+1:r].copy()
    if side.find("l") > -1:
        xint = np.concatenate((xl, xint))
        yint = np.concatenate((yl, yint))
    if side.find("r") > -1:
        xint = np.concatenate((xint, xr))
        yint = np.concatenate((yint, yr))
    #IUS
    yint2 = interpolate(xint, yint, k = 5, ext=3)

    return yint2

class SphericalBesselTransform:
    '''
    Spherical bessel function class with FFTW option added. Based on Stephen Chen's velocileptor's (https://github.com/sfschen/velocileptors) version of Chirag Modi's CLEFT code (https://github.com/modichirag/CLEFT).

    Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
    the untransformed coordinate, up to a given order L in bessel functions (j_l for l
    less than or equal to L. The point is to save time by evaluating the Mellin transforms
    u_m in advance.

    Uses pyfftw, which can perform multiple (ncol) Fourier transforms at once, one for
    each bias contribution.

    Based on Yin Li's package mcfit (https://github.com/eelregit/mcfit)
    with the above modifications.

    '''

    def __init__(self, qs, L=15, ncol = 1, low_ring=True, fourier=False, threads=1,
                 import_wisdom=False, wisdom_file='./fftw_wisdom.npy',useFFTW=True):
        '''
        Parameters:
        ---------
        qs: array,float
            Wavenumber abscissa pts
        L: int,optional
            Maximum order in \ell used for the angular sum used to compute P_ZA as 1d integrals (Schneider & Bartelmann 1995)
        ncol: int,optional
            Used to compute multiple transforms at once for different biasing terms, for za will be 1
        low_ring: boolean, optional
            Whether to apply lowring condition (force continuity when folding period, https://jila.colorado.edu/~ajsh/FFTLog/#lowring)
        fourier: boolean,optional
            If true assume pi factor for an integral in fourier space
        import_wisdom: boolean, optional
            Whether to use saved setup for the fftw to reduce overhead
        wisdom_file: string, optional
            The filename for the above
        useFFTW: boolean, optional
            Whether to use pyfftw (default) or numpy ffts
        '''
        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = np.sqrt(np.pi)
        else:
            self.sqrtpi = np.sqrt(np.pi) / (2*np.pi**2)

        self.q = qs
        self.L = L
        self.ncol = ncol

        self.Nx = len(qs)
        self.Delta = np.log(qs[-1]/qs[0])/(self.Nx-1)
        self.useFFTW = useFFTW

        # zero pad the arrays to the preferred length format for ffts, 2^N
        self.N = 2**(int(np.ceil(np.log2(self.Nx))) + 1)
        self.Npad = self.N - self.Nx
        if(self.useFFTW): #fftw setup
            self.ii_l = self.Npad - self.Npad//2 # left and right indices sandwiching the padding
            self.ii_r = self.N - self.Npad//2
                    # Set up FFTW objects:
            if import_wisdom:
                pyfftw.import_wisdom(tuple(np.load(wisdom_file)))

            self.fks = pyfftw.empty_aligned((self.ncol,self.N//2 + 1), dtype='complex128')
            self.fs  = pyfftw.empty_aligned((self.ncol,self.N), dtype='float64')

            self.gks = pyfftw.empty_aligned((self.ncol,self.N//2 + 1), dtype='complex128')
            self.gs  = pyfftw.empty_aligned((self.ncol,self.N), dtype='float64')

            pyfftw.config.NUM_THREADS = threads
            self.fft_object = pyfftw.FFTW(self.fs, self.fks, direction='FFTW_FORWARD',threads=threads)
            self.ifft_object = pyfftw.FFTW(self.gks, self.gs, direction='FFTW_BACKWARD',threads=threads)
            # Set up FFTW objects:
            if import_wisdom:
                pyfftw.import_wisdom(tuple(np.load(wisdom_file)))
        else:
            self.pads = np.zeros( (self.N-self.Nx)//2  )
            self.pad_iis = np.arange(self.Npad - self.Npad//2, self.N - self.Npad//2)

        # Set up the FFTLog kernels u_m up to, but not including, L
        ms = np.arange(0, self.N//2+1)
        self.ydict = {}; self.udict = {}; self.qdict= {}

        #L is the number of orders to track through for bessel functions
        #Recall that lowringing is setting the bdr values of lnxy to be equal so that the period folds continuously
        #Here this is just done L times because we will need that many bessel kernels in the angular integration
        if low_ring:
            for ll in range(L):
                q = max(0, 1.5 - ll)
                lnxy = self.Delta/np.pi * np.angle(self.UK(ll,q+1j*np.pi/self.Delta)) #ln(xmin*ymax)
                ys = np.exp( lnxy - self.Delta) * qs/ (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) \
                        * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)
                if(self.useFFTW): us[self.N//2] = us[self.N//2].real # manually impose low ring

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q

        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(L):
                q = max(0, 1.5 - ll)
                ys = np.exp(-self.Delta) * qs / (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms)
                if(self.useFFTW): us[self.N//2] = us[self.N//2].real # manually impose low ring

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q


    def export_wisdom(self, wisdom_file='./fftw_wisdom.npy'):
        '''Parameters:
         ----------
         wisdom_file: string,optional
            Wisdom filename
         '''
        np.save(wisdom_file, pyfftw.export_wisdom())

    def sph(self, nu, fq):
        '''
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates self.q.
        Parameters:
        ----------
        nu: int
            Order of spherical bessel function
        fq: array (float)
            Values of function to transform
        Returns:
        ----------
        (array (float), array (float))
            Tuple of the (absissca, function) values in the transformed space
        '''
        q = self.qdict[nu]; y = self.ydict[nu] #pre-tabulated for nus we want
        useFFTW = self.useFFTW
        if(useFFTW):
            self.fs[:] = 0 # on NERSC this seems necessary or this variable spills over from previous calls
            self.fs[:,self.Npad - self.Npad//2 : self.N - self.Npad//2] = fq * self.q**(3-q)
            fks = self.fft_object()
            self.gks[:] = np.conj(fks * self.udict[nu])
            gs = self.ifft_object()
            return y, gs[:,self.ii_l:self.ii_r] * y**(-q)
        else:
            f = np.concatenate( (self.pads,self.q**(3-q)*fq,self.pads) )
            fks = np.fft.rfft(f)
            gks = self.udict[nu] * fks
            gs = np.fft.hfft(gks) / self.N
            return y, y**(-q) * gs[self.pad_iis]

    def UK(self, nu, z):
        '''
        The Mellin transform of the spherical bessel transform. - from mcfit
        Parameters:
        ----------
        nu: int
            Order of spherical bessel function
        z: float
            Argument
        Returns:
        ----------
        array (float)
            Spherical bessel kernel evaluated at z
        '''
        return self.sqrtpi * np.exp(np.log(2)*(z-2) + loggamma(0.5*(nu+z)) - loggamma(0.5*(3+nu-z)))

#Qfunc for computing the power - tossing almost everything from CLEFT
class QFuncFFT:
    '''
       Chirag's notes:
       Class to calculate all the functions of q, X(q), Y(q), U(q), xi(q) etc.
       as well as the one-loop terms Q_n(k), R_n(k) using FFTLog.
       Throughout we use the ``generalized correlation function'' notation of 1603.04405.
       Note that one should always cut off the input power spectrum above some scale.
       I use exp(- (k/20)^2 ) but a cutoff at scales twice smaller works equivalently,
       and probably beyond that. The important thing is to keep all integrals finite.
       This is done automatically in the Zeldovich class.
       Currently using the numpy version of fft. The FFTW takes longer to start up and
       the resulting speedup is unnecessary in this case.

    '''
    def __init__(self, k, p, qv = None, useFFTW = True):
        '''This assumes L=5 (with lowring) for the angular evaluation
        Parameters:
        ----------
        k: array (float)
            Wavenumbers
        p: array (float)
            Power spectrum values
        qv: array (float),optional
            q values to use when computing the generalized correlation functions
        '''
        self.k = k
        self.p = p
        if qv is None:
            self.qv = np.logspace(-5,5,20000) #can decrease?
        else:
            self.qv = qv

        self.sph = SphericalBesselTransform(self.k, L=5, low_ring=True, fourier=True,
                                            useFFTW=useFFTW) #keeping with old choice
        self.setup_xiln()
        self.setup_2pts()

    def setup_xiln(self):
        # Compute a bunch of generalized correlation functions
        # self.xi00 = self.xi_l_n(0,0)
        # self.xi1m1 = self.xi_l_n(1,-1)
        self.xi0m2 = self.xi_l_n(0,-2, side='right') # since this approaches constant on the left only interpolate on right
        self.xi2m2 = self.xi_l_n(2,-2)

    def setup_2pts(self):
        # Piece together xi_l_n into what we need
        self.Xlin = 2./3 * (self.xi0m2[0] - self.xi0m2 - self.xi2m2)
        self.Ylin = 2 * self.xi2m2
        # self.Ulin = - self.xi1m1
        # self.corlin = self.xi00

    def xi_l_n(self, l, n, _int=None, extrap=False, qmin=1e-3, qmax=1000, side='both'):
        '''
        Calculates the generalized correlation function xi_l_n (Schmittful et al 16), which is xi when l = n = 0
        If _int is None assume integrating the power spectrum.
        Parameters:
        ----------
        l: int
            Legendre expansion order
        n: int
            Power on q in the integrand
        Returns:
        ----------
        array (float)
            Xi_l_n evaluated at qv
        '''
        if _int is None:
            integrand = self.p * self.k**n
        else:
            integrand = _int * self.k**n

        qs, xint =  self.sph.sph(l,integrand)

        if extrap:
            qrange = (qs > qmin) * (qs < qmax)
            return loginterp(qs[qrange],xint[qrange],side=side)(self.qv)
        else:
            return np.interp(self.qv, qs, xint)

#Stripped down CLEFT class
class CLEFT:
    '''
    jms - Removing everything but linear
    Class to calculate power spectra up to one loop.
    Based on Chirag's code
    https://github.com/sfschen/velocileptors/blob/master/LPT/cleft_fftw.py
    The bias parameters are ordered in pktable as 1
    '''

    def __init__(self, k, p, cutoff=10, jn=5, N = 2000, threads=1, extrap_min = -5,
                 extrap_max = 3, import_wisdom=False, wisdom_file='wisdom.npy',
                 useFFTW=True):
        '''
        Parameters:
        ----------
        k: array (float)
            Wavenumber array
        p: array (float)
            Power array
        cutoff: float
            k-scale of exponential suppression to allow for well-behaved FFT
        jn: int,optional
            Maximum order in \ell used for the angular sum used to compute P_ZA as 1d integrals (Schneider & Bartelmann 1995)
        N: int,optional
            Number of points used in extrap abscissa array
        threads: int,optional
            Num threads passed to pyfftw
        extrap_min: int,optional
            Min in logk used in extrap array
        extrap_max: int,optional
            Max in logk used in extrap array
        import_wisdom: boolean, optional
            Whether to use saved setup for the fftw to reduce overhead
        wisdom_file: string, optional
            The filename for the above

        '''
        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min

        self.cutoff = cutoff
        self.kint = np.logspace(extrap_min,extrap_max,self.N)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)
        self.useFFTW = useFFTW

        self.update_power_spectrum(k,p)

        self.pktable = None
        self.num_power_components = 1 #since just ZA
        self.nkt = 100

        self.jn = jn
        self.threads = threads
        self.import_wisdom = import_wisdom
        self.wisdom_file = wisdom_file
        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.nkt,#self.num_power_components,
                                            threads=self.threads, import_wisdom= self.import_wisdom,
                                            wisdom_file = self.wisdom_file, useFFTW=self.useFFTW)
        #uses FFTW version by default

    def update_power_spectrum(self, k, p):
        # Updates the power spectrum and various q functions. Can continually compute for new cosmologies without reloading FFTW
        self.k = k
        self.p = p
        self.pint = loginterp(k,p)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.setup_powerspectrum()

    def setup_powerspectrum(self):
        # This sets up terms up to one looop in the combination (symmetry factors) they appear in pk
        self.qf = QFuncFFT(self.kint, self.pint, qv=self.qint,useFFTW=False)

        # linear terms
        self.Xlin = self.qf.Xlin
        self.Ylin = self.qf.Ylin

        self.XYlin = self.Xlin + self.Ylin; self.sigma = self.XYlin[-1]
        self.yq = self.Ylin / self.qint
        # self.corlin = self.qf.corlin

    def p_integrals(self, k, Dz=None):
        '''
        Compute P(k) for a single k as a vector of all bias contributions.
        Parameters:
        ----------
        k: array (float)
            Wavenumber array
        Returns:
        ----------
        array
            ZA power
        '''
        Dz = Dz if(Dz is not None) else 1.
        ksq = k**2
        nk=len(k) if type(k) is not np.float64 else 1
        expon = np.exp( -0.5*ksq.reshape([nk,1]) * (self.XYlin.reshape([1,len(self.XYlin)]) - self.sigma) * Dz**2 )
        suppress = np.exp( -0.5 * ksq.reshape([nk,1]) *self.sigma  * Dz**2 ) #size k


        # ^try setting to size (k,q) does sph work on 2d arr? looks like np version will
        # one shot at this then use veloc

        # ret = np.zeros((self.num_power_components,len(k)))
        ret = np.zeros((nk,1)) if nk>1 else 0.
        # print(nk,expon.shape)

        #if want to get fancy can put z dim in front...
        bias_integrands = np.zeros( (self.nkt,self.N)  ) if nk>1 else np.zeros((1,self.N))

        for l in range(self.jn):
            # bias_integrands[-1,:] = 1 # this is the counterterm, minus a factor of k2
            bias_integrands[:,:] = 1 # this is the counterterm, minus a factor of k2

            #this is the ZA
            # multiply by IR exponent
            # ahh so this is bad, before this wouldn't everybody to zero , we want this to be ones
            if l == 0:
                bias_integrands = bias_integrands * expon
                # print('what is this:',bias_integrands[:,-1][:,None])
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * (self.yq * Dz**2)**l

            ktemps, bias_ffts = self.sph.sph(l, bias_integrands) #returns size kint array
            ret[:,0] +=  k**l * np.array([interp1d(ktemps, bias_ffts[i])(k[i]) for i in range(nk)])
        return 4*suppress*np.pi*ret #for all k should return (nktarget)

    def make_ptable(self, kmin = 1e-3, kmax = 3, nk = 100, Dz=None):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        Parameters:
        ----------
        kmin: float,optional
            Min wavenumber
        kmax: float,optional
            Max wavenumber
        nk: int,optional
            Number of pts
        Returns:
        ----------
        None
        '''
        self.pktable = np.zeros([nk, 1+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable[:, 0] = kv[:]
        self.pktable[:, 1:] = self.p_integrals(kv,Dz=Dz)


    def export_wisdom(self, wisdom_file='./wisdom.npy'):
        """
        Wrapper for the SBT class wisdom saving function
        Parameters:
        ----------
        wisdom_file: string,optional
            Filename for wisdom.
        """
        self.sph.export_wisdom(wisdom_file=wisdom_file)

    #From Stephen's gsm model class, appropriated here for xiza since it is all I want
    def compute_xi_real(self, rr, kswitch=1e-2,kint= np.logspace(-5,3,4000)):
        '''
        Compute the real-space correlation function at rr.
        '''
        #kint = np.logspace(-5,3,4000)
        plin = loginterp(self.k, self.p)(kint)
        # This is just the zeroth moment:
        kv   = self.pktable[:,0]
        pzel = self.pktable[:,-1]

        peft = pzel
        weight =  0.5 * (1 + np.tanh(3*np.log(kint/kswitch))) #smooth away low-k to replace with plin below
        weight[weight < 1e-3] = 0

        _integrand = loginterp(kv, peft)(kint) #at some point turn into theory extrapolation
        _integrand = weight * _integrand + (1-weight) * (plin)

        sph_xi = SphericalBesselTransform(kint,L=1,fourier=True,
                                            useFFTW=self.useFFTW) #keeping with old choice, for whatever reason
        #dropping L to 1 since we won't be needing higher here for linear, only using sph for j0 transform to xi

        qint, xi = sph_xi.sph(0,_integrand)

        xir = interpolate(qint,xi)(rr)
        return xir
