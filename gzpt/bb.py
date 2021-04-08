import numpy as np
"The broadband beyond Zeldovich function"

def PBB(k,params,nmax,wantGrad=False):
    '''Pade Power
    Parameters:
    ----------
    k: array
        wavenumbers
    params: list
        hzpt parameters, floats
    nmax: int, optional
        order of Pade expansion
    wantGrad: boolean, optional
        whether to return function or (function,gradient)
    Returns:
    ---------
    array (of size k)
        broadband function value
    tuple of arrays, optional
        broadband function value and gradient wrt hzpt parameters.

    '''

    if(nmax==0):
        A0,R = params
        R1h,R1sq,R2h = None, None, None
    elif(nmax==1):
        A0,R,R1h = params
        R1sq,R2h = None, None
    elif(nmax==2):
        A0,R,R1h,R1sq,R12 = params
        R2h = R1h/(np.sqrt(2)*R12) #R12 is R1h/R2h/sqrt(2) and is forced to be > 1
        if(R12<=.99) : raise ValueError("R12 must be greater than 1 for physical values.")

    def F_comp(k,R):
        """For now only Lorentzian compensation"""
        Fk = 1-1/(1+(k*R)**2)
        return Fk

    def pade(k,R1h,R1sq,R2h):
        if(nmax==0):
            return 1.
        if(nmax==1):
            return 1/(1+(k*R1h)**2)
        elif(nmax==2):
            return (1+ R1sq * k**2) /(1+(k*R1h)**2 + (k*R2h)**4)


    def gradient(k,A0,R,R1h,R1sq,R2h):
        A_grad = F_comp(k,R) * pade(k,R1h,R1sq,R2h)
        if(nmax==0):
            R_grad = (2*A0* k**2)/(R + k**2 * R**3) * F_comp(k,R)
            return np.array([A_grad,R_grad])
        if(nmax==1):
            R_grad = (2*A0)/((1 + k**2 * R1h**2)* k**2 * R**3) * F_comp(k,R)**2
            R1h_grad = -(2*A0* k**2 *R1h)/(1 + k**2 * R1h**2)**2 * F_comp(k,R)
            return np.array([A_grad,R_grad,R1h_grad])
        elif(nmax==2):
            k2,k4 = k**2,k**4
            R1h2,R2h4 = R1h**2,R2h**4
            R_grad = 2*A0*(1+ k2 * R1sq)/((1 + k2 * R1h2 + k4 * R2h4)* k2 * R**3) * F_comp(k,R)**2
            R1h_num = -2*A0* (k2 + 2*(R2h4/R1h2)*k4) * R1h * (1 + k2 * R1sq ) * F_comp(k,R)
            R1h_denom = (1 + k2 * R1h2 + k4 * R2h4)**2
            R1h_grad = R1h_num/R1h_denom
            R1sq_grad = (A0* k2)/(1 + k2 * R1h2 +k4 * R2h4) * F_comp(k,R)
            R2h_grad = -4*A0* k4 * (1+k2 * R1sq) *R2h**3 /(1 + k2 * R1h2 + k4 *R2h4)**2 * F_comp(k,R)

            #convert R2h_grad to R12_grad - R12 = \frac{R1h}{R2h} => \frac{}\partial(PBB(...,R2h))}{\partial{R12}} = grad_R2h * \frac{\partial{R2h}}{\partial{R12}}
            #this last factor is chain_fac and is just \frac{-1*R1h}{sqrt{2} R12^2} = \frac{-sqr{2}*R2h^2}{R1h}
            chain_fac = -np.sqrt(2)*R2h**2 /R1h
            R12_grad = R2h_grad*chain_fac
            #d(pbb)/d(R1h) = A0 F(k) d(bb)/dR1h, d(bb)/dR1h = (1+R1sq k^2) d((1 + k^2 R1h^2 + k^4 R2h^4)^(-1))/d(R1h)

            return np.array([A_grad,R_grad,R1h_grad,R1sq_grad,R12_grad])


    Pbb = A0 * F_comp(k,R) * pade(k,R1h,R1sq,R2h)

    if(wantGrad==True):
        return Pbb, gradient(k,A0,R,R1h,R1sq,R2h).T
    else:
        return Pbb


def XiBB(r,params,nmax,wantGrad=False):
    '''Pade Correlation Function
    Parameters:
    ----------
    r: array
        scale
    params: list
        hzpt parameters, floats
    nmax: int, optional
        order of Pade expansion
    wantGrad: boolean, optional
        whether to return function or (function,grad)
    Returns:
    ---------
    array (of size k)
        broadband function value
    tuple of arrays, optional
        broadband function value and gradient wrt hzpt parameters.

    '''

    if(nmax==0):
        A0,R = params
        R1h,R1sq,R2h=None,None,None
    elif(nmax==1):
        A0,R,R1h = params
        R1sq,R2h = None, None
    elif(nmax==2):
        A0,R,R1h,R1sq,R12 = params
        R2h = R1h/(np.sqrt(2)*R12) #R12 is R1h/R2h/sqrt(2) and is forced to be > 1
        if(R12<=.99) : raise ValueError("R12 must be greater than 1. for physical values.")
    else:
        raise NotImplementedError("nmax>2 not supported for xi")

    def F2_comp(r,R):
        '''Exponential `compensation' - analytic fourier transform of F_comp'''
        F2 = np.exp(-r/R) / (4*np.pi*r*R**2)
        return F2


    def pade2(r,R,R1h,R1sq,R2h):
        if(nmax==0):
            return 1
        elif(nmax==1):
            return (1 - (R/R1h)**2 * np.exp(-(R-R1h)*r/(R*R1h)))/(1 - (R1h/R)**2)
        elif(nmax==2):
            '''Hand et al. 2017 eqns. (B.8-B.11) '''
            pre = 1/(1 -(R1h/R)**2 +(R2h/R)**4)
            S = np.sqrt(R1h**4 - 4*R2h**4)
            A = (1/(2*R2h**4 *S))*(R**2 *(-2*R2h**4 +R1sq *(R1h**2 -S)) + R2h**4 *(R1h**2 -S) + R1sq *(-R1h**4 +2*R2h**4 + R1h**2 *S))
            B = -(1/(2*R2h**4 *S))*(R2h**4 *(R1h**2 + S) - R1sq *(R1h**4 - 2*R2h**4 +R1h**2 *S) + R**2 *(-2*R2h**4 + R1sq *(R1h**2 + S)))

            first = 1- R1sq*(1/R)**2
            second = A*np.exp(r*(1/R - (1/R2h**2)*np.sqrt((R1h**2 - S)/2)))
            third = B*np.exp(r*(1/R - (1/R2h**2)*np.sqrt((R1h**2 + S)/2)))

            return pre*(first + second + third)

    def gradient(r,A0,R,R1h,R1sq,R2h):
        A_grad = - F2_comp(r,R) * pade2(r,R,R1h,R1sq,R2h)
        F_comp_R_grad = -A0 * (r - 2.*R)/R**2 *F2_comp(r,R)

        if(nmax==0):
            R_grad = F_comp_grad
            return np.array([A_grad,R_grad])
        if(nmax==1):
            R2,R1h2,R1h4 = R**2,R1h**2,R1h**4
            R_num = -A0*np.exp(-r/R)*(2.*R**3 *(-1. + np.exp(r*(1/R - 1/R1h)))
                                        +r*(R2 - R1h2)
                                       )
            R_denom = 4.*np.pi*r*(R**3 - R*R1h2)**2
            R_grad = R_num/R_denom
            R1h_num = A0*np.exp(-r/R)*(-2. * R1h**5
                                        + np.exp(r*(1/R - 1/R1h))*R2 *(-2.*R2 *R1h
                                                                       +4.*R1h**3
                                                                       +r*(R2 - R1h2)
                                                                       )
                                        )
            R1h_denom = 4.*np.pi*r*R1h4 * (R2 - R1h2)**2
            R1h_grad = R1h_num/R1h_denom
            return np.array([A_grad,R_grad,R1h_grad])
        elif(nmax==2):
            "Chain rule for this very long expression"
            """TODO:- should make it so don't need to run this twice - probably refactor the gradient into the actual function, turn into f+j"""
            #setup
            R2,R3,R4,R5,R6 = R**2, R**3, R**4, R**5,R**6
            R1h2,R1h3,R1h4,R1h5,R1h6,R1h7,R1h8 = R1h**2, R1h**3, R1h**4, R1h**5, R1h**6, R1h**7, R1h**8
            R2h2,R2h3,R2h4,R2h5,R2h8 = R2h**2, R2h**3, R2h**4, R2h**5, R2h**8

            pre = 1/(1 -(R1h2/R2) +(R2h4/R4))
            S = np.sqrt(R1h4 - 4*R2h4)
            A = (1/(2*R2h4 *S))*(R2 *(-2*R2h4 +R1sq *(R1h2 -S)) + R2h4 *(R1h2 -S) + R1sq *(-R1h4 +2*R2h4 + R1h2 *S))
            B = -(1/(2*R2h4 *S))*(R2h4 *(R1h**2 + S) - R1sq *(R1h4 - 2*R2h4 +R1h2 *S) + R2 *(-2*R2h4 + R1sq *(R1h2 + S)))

            T1 = 1- R1sq/R2
            EA = np.exp(r*(1/R - (1/R2h2)*np.sqrt((R1h2 - S)/2)))
            EB = np.exp(r*(1/R - (1/R2h2)*np.sqrt((R1h2 + S)/2)))
            T2 = A*EA
            T3 = B*EB

            #R
            pre_R_grad = (-2.*R5*R1h2 + 4*R3*R2h4)/(R4 - R2*R1h2 + R2h4)**2
            T1_R_grad = 2*R1sq/R3
            A_R_grad = R*(R1h2*R1sq -2.*R2h4 -R1sq*S)/(R2h4*S)
            EA_R_grad = -(r/R2)*EA
            B_R_grad = -R*(-2.*R2h4 + R1sq*(R1h2 + S))/(R2h4*S)
            EB_R_grad = -(r/R2)*EB
            T2_R_grad = A_R_grad*EA + A*EA_R_grad
            T3_R_grad = B_R_grad*EB + B*EB_R_grad
            BB_R_grad = pre_R_grad*(T1+T2+T3) + pre*(T1_R_grad + T2_R_grad + T3_R_grad)

            R_grad = F_comp_R_grad*pade2(r,R,R1h,R1sq,R2h) + (- A0 * F2_comp(r,R))*BB_R_grad

            #R1h
            pre_R1h_grad = (2.*R6*R1h2-4.*R4*R2h4)/(R1h*(R4 - R2*R1h2 + R2h4)**2)
            S_R1h_grad = 2.*S/R1h
            A_R1h_grad = (R1sq*(R1h4 - 2.*R2h4 - R1h2*S)
                          + 2.*R2*(-R1h2*R1sq + R2h4 + R1sq*S))/(R1h*R2h4*S)
            EA_R1h_grad = ((r*np.sqrt((R1h2-S)/2.))/(R1h*R2h2)) * EA
            B_R1h_grad = (-R1sq*(R1h4 - 2.*R2h4 + R1h2*S)
                           + 2*R2*(-R2h4 + R1sq*(R1h2 + S)))/(R1h*R2h4*S)
            EB_R1h_grad = ((r*np.sqrt((R1h2+S)/2.))/(R1h*R2h2)) * EB
            T2_R1h_grad = A_R1h_grad*EA + A*EA_R1h_grad
            T3_R1h_grad = B_R1h_grad*EB + B*EB_R1h_grad
            BB_R1h_grad = pre_R1h_grad*(T1+T2+T3) + pre*(T2_R1h_grad + T3_R1h_grad)

            R1h_grad = (-A0*F2_comp(r,R))*BB_R1h_grad

            #R1sq
            T1_R1sq_grad = -1./R2
            A_R1sq_grad = (2.*R2h4 + (R-R1h)*(R+R1h)*(R1h2-S))/(2*R2h4*S)
            B_R1sq_grad = -(2.*R2h4 + (R-R1h)*(R+R1h)*(R1h2+S))/(2*R2h4*S)
            T2_R1sq_grad = A_R1sq_grad*EA
            T3_R1sq_grad = B_R1sq_grad*EB
            BB_R1sq_grad = pre*(T1_R1sq_grad + T2_R1sq_grad + T3_R1sq_grad)

            R1sq_grad = (-A0*F2_comp(r,R))*BB_R1sq_grad

            #R2h
            pre_R2h_grad = -(4.*R4*R2h3)/(R4 - R2*R1h2 + R2h4)**2
            S_R2h_grad = -8.*R2h3/S
            A_R2h_grad = -2.*(-R1h8*R1sq + 6.*R1h4*R1sq*R2h4 - 4.*R1sq*R2h8
                              + R1h6*R1sq*S - 2.*R1h2*(R2h8 + 2.*R1sq*R2h4*S)
                              + R2*(R1h6*R1sq - 6.*R1h2*R1sq*R2h4 - R1h4*R1sq*S
                                    + 4.*(R2h8 + R1sq*R2h4*S)
                                   )
                             )/(R2h5*(S**3))
            EA_R2h_grad = r*(-R1h4 + 2.*R2h4 + R1h2*S)/(R2h3*S*np.sqrt((R1h2-S)/2.)) * EA
            B_R2h_grad = -2.*(R1h8*R1sq - 6.*R1h4*R1sq*R2h4 + 4.*R1sq*R2h8
                              + R1h6*R1sq*S - R2*(R1h6*R1sq - 6.*R1h2*R1sq*R2h4
                                                  + 4.*R2h8 + R1h4*R1sq*S -4.*R1sq*R2h4*S
                                                 )
                              + 2.*R1h2*(R2h8 - 2.*R1sq*R2h4*S)
                             )/(R2h5*(S**3))
            EB_R2h_grad = r*(R1h4 - 2.*R2h4 + R1h2*S)/(R2h3*S*np.sqrt((R1h2+S)/2.)) * EB
            T2_R2h_grad = A_R2h_grad*EA + A*EA_R2h_grad
            T3_R2h_grad = B_R2h_grad*EB + B*EB_R2h_grad
            BB_R2h_grad = pre_R2h_grad*(T1+T2+T3) + pre*(T2_R2h_grad + T3_R2h_grad)

            R2h_grad = (-A0*F2_comp(r,R))*BB_R2h_grad

            #see note for description of chain factor in PBB
            chain_fac = -np.sqrt(2)*R2h**2 /R1h
            R12_grad = R2h_grad*chain_fac

            return np.array([A_grad,R_grad,R1h_grad,R1sq_grad,R12_grad])

    Xibb = -A0 * F2_comp(r,R) * pade2(r,R,R1h,R1sq,R2h)
    if(wantGrad==True):
        return Xibb, gradient(r,A0,R,R1h,R1sq,R2h).T
    else:
        return Xibb
