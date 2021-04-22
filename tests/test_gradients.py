"Test analytic gradients against finite differences (to check for mistakes)"
import gzpt
from gzpt import zel,matter,tracers
import numpy as np
import os

def test_gradients():
    #arbitrarily select reasonable parameter values

    #finite difference
    atol,rtol,eps=1e-4,1e-8,1e-8

    #set up the model
    testfile = os.path.join(os.path.dirname(__file__), './test_plin_cc_z0.55.txt')
    k,plin = np.loadtxt(testfile,unpack=True)
    z=0.55
    model = gzpt.hzpt(k,plin)
    ktest = np.logspace(-3,1,1000)
    rtest = np.logspace(-1,3,1000)

    #BB
    mm_ps = np.array([350., 26., 5.5, 15., 1.9])

    #nmax=1
    mm = matter.Correlator(mm_ps[:-2],model)
    ind = np.ones(3)
    gradsk1 = mm.Power(wantGrad=True)(ktest)[1]
    gradsr1 = mm.Xi(wantGrad=True)(rtest)[1]
    for i in range(3):
        indp,indm = np.copy(ind),np.copy(ind)
        indp[i]+=eps
        indm[i]-=eps
        mplu = (matter.Correlator(mm_ps[:-2]*indp ,model))
        mmin = (matter.Correlator(mm_ps[:-2]*indm ,model))
        delta = (mplu.Power()(ktest)-mmin.Power()(ktest))
        assert np.allclose((delta)/(2*eps*mm_ps[:-2][i]),gradsk1[:,i],rtol=rtol,atol=atol), \
        "FS nmax=1 gradient test failed!"
        #CS
        mplu = (matter.Correlator(mm_ps[:-2]*indp ,model))
        mmin = (matter.Correlator(mm_ps[:-2]*indm ,model))
        delta = (mplu.Xi()(rtest)-mmin.Xi()(rtest))
        assert np.allclose((delta)/(2*eps*mm_ps[:-2][i]),gradsr1[:,i],rtol=rtol,atol=atol), \
        "CS nmax=1 gradient test failed!"

    #nmax=2
    ind = np.ones(5)
    mm = matter.Correlator(mm_ps,model)
    gradsk = mm.Power(wantGrad=True)(ktest)[1]
    gradsr = mm.Xi(wantGrad=True)(rtest)[1]
    for i in range(5):
        indp,indm = np.copy(ind),np.copy(ind)
        indp[i]+=eps
        indm[i]-=eps
        #FS
        mplu = (matter.Correlator(mm_ps*indp ,model))
        mmin = (matter.Correlator(mm_ps*indm ,model))
        delta = (mplu.Power()(ktest)-mmin.Power()(ktest))
        assert np.allclose((delta)/(2*eps*mm_ps[i]),gradsk[:,i],rtol=rtol,atol=atol), \
        "FS nmax=2 gradient test failed!"
        #CS
        mplu = (matter.Correlator(mm_ps*indp ,model))
        mmin = (matter.Correlator(mm_ps*indm ,model))
        delta = (mplu.Xi()(rtest)-mmin.Xi()(rtest))
        assert np.allclose((delta)/(2*eps*mm_ps[i]),gradsr[:,i],rtol=rtol,atol=atol), \
        "CS nmax=2 gradient test failed!"

    #tracer-tracer - just get the bias, exclusion, sat radii
    test_tt_params = [[1e-4,2.,350., 26., 5.5], [2., .1], [1e3, 1]]
    tt = tracers.AutoCorrelator(test_tt_params[0],model,
                                params_exc=test_tt_params[1],params_sat=test_tt_params[1])
    xitt_test,xitt_grad = tt.Xi(wantGrad=True)(rtest)
    Ptt_test,Ptt_grad = tt.Xi(wantGrad=True)(ktest)
    assert ~np.any(np.isnan(Ptt_grad)), \
    "tracer.AutoCorrelator.Power gradient does not exist"
    assert ~np.any(np.isnan(xitt_grad)), \
    "tracer.AutoCorrelator.Xi gradient does not exist"
    print('Done')
    # #projected statistics - WIP
    # rpwp,wpgg,wpgg_grad = tt.wp(rtest,wantGrad=True)
    # #made up test values for DS
    # rpDS,DSgm,DSgm_grad = tm.Delta_Sigma(rtest,12,1e3,.8e3,1,3e11,wantGrad=True)
    # assert ~np.any(np.isnan(wpgg_grad)), \
    # "tracer.AutoCorrelator.wp gradient does not exist"
    # assert ~np.any(np.isnan(DSgm_grad)), \
    # "tracer.CrossCorrelator.DS gradient does not exist"

    return None
