"Simple check to make sure gradients exist"
import gzpt
from gzpt import zel,matter,tracers
import numpy as np
import os

def test_gradients_exist():
    #arbitrarily select reasonable parameter values

    #set up the model
    testfile = os.path.join(os.path.dirname(__file__), './test_plin_planck_z0.55.txt')
    k,plin = np.loadtxt(testfile,unpack=True)
    z=0.55
    model = gzpt.hzpt(k,plin,z)
    ktest = np.logspace(-3,1,1000)
    rtest = np.logspace(-1,3,1000)

    #matter
    test_mm_params = np.array([350., 26., 5.5, 15., 1.9])
    mm = matter.Correlator(test_mm_params,model)
    Pmm_test,Pmm_grad = mm.Power(wantGrad=True)(ktest)
    ximm_test,ximm_grad = mm.Xi(wantGrad=True)(rtest)
    assert ~np.any(np.isnan(Pmm_grad)), \
    "matter.Power gradient does not exist"
    assert ~np.any(np.isnan(ximm_grad)), \
    "matter.Xi gradient does not exist"

    #tracer-matter
    test_tm_params = np.array([2.,350., 26., 5.5, 15., 1.9])
    tm = tracers.CrossCorrelator(test_tm_params,model)
    Ptm_test,Ptm_grad = tm.Power(wantGrad=True)(ktest)
    xitm_test,xitm_grad = tm.Xi(wantGrad=True)(rtest)
    assert ~np.any(np.isnan(Ptm_grad)), \
    "tracer.CrossCorrelator.Power gradient does not exist"
    assert ~np.any(np.isnan(xitm_grad)), \
    "tracer.CrossCorrelator.Xi gradient does not exist"

    #tracer-tracer
    test_tt_params = np.array([1e-4,2.,350., 26., 5.5, 2., .1, 1e3, 1])
    ttno = tracers.AutoCorrelator(test_tt_params[:5],model,useExc=False)
    tt = tracers.AutoCorrelator(test_tt_params,model,useExc=True)
    Ptt_test,Ptt_grad = ttno.Power(wantGrad=True)(ktest)
    xitt_test,xitt_grad = tt.Xi(wantGrad=True)(rtest)
    assert ~np.any(np.isnan(Ptt_grad)), \
    "tracer.AutoCorrelator.Power gradient does not exist"
    assert ~np.any(np.isnan(xitt_grad)), \
    "tracer.AutoCorrelator.Xi gradient does not exist"

    #projected statistics
    rpwp,wpgg,wpgg_grad = ttno.wp(rtest,wantGrad=True)
    #made up test values for DS
    rpDS,DSgm,DSgm_grad = tm.Delta_Sigma(rtest,12,1e3,.8e3,1,3e11,wantGrad=True)
    assert ~np.any(np.isnan(wpgg_grad)), \
    "tracer.AutoCorrelator.wp gradient does not exist"
    assert ~np.any(np.isnan(DSgm_grad)), \
    "tracer.CrossCorrelator.DS gradient does not exist"

    return None
