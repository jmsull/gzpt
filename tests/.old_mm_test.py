"Test harmonic equivalence of correlation function and power spectrum w/ pyfftw."
import gzpt
from gzpt import zel,matter
from gzpt.zel import loginterp
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius

def test_pmm_ximm_params():
#arbitrarily select reasonable parameter values
    test_params = np.array([350., 26., 5.5, 15., 1.9])

#set up the model
    k,plin = np.loadtxt('./test_plin_planck_z0.55.txt',unpack=True)
    z=0.55
    model = gzpt.hzpt(k,plin,z)
    kint=np.logspace(-3,1,4000)
    mm = matter.Correlator(test_params,model)
    print(mm,type(mm))
#fix the power spectrum
    P_test = mm.Power()(kint)
#take numerical fourier transform
    weight =  0.5 * (1 + np.tanh(3*np.log(kint/1e-2)))
    weight[weight < 1e-3] = 0
    smoothed = weight * P_test + (1-weight) * (model.plin(kint))
    r_num,xi_num_s = zel.SphericalBesselTransform(kint,L=1,fourier=True,useFFTW=False).sph(0,smoothed)
#compute transform of analytic expression at the same parameter values
    rtest = np.logspace(np.log10(1/k.max()),np.log10(1/k.min()),len(k))
    xi_exact = mm.Xi()(rtest)
    xi_num = ius(r_num,xi_num_s)(rtest)
    print(r_num)
    print(rtest)
    print(xi_num/xi_exact)
    assert np.allclose(xi_num,xi_exact,rtol=.01) #high tolerance but do not trust model to better than this
    return None
