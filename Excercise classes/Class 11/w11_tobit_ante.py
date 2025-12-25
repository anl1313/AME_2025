import numpy as np 
from scipy.stats import norm
from numpy import linalg as la

name = 'Tobit'

def q(theta, y, x): 
    return None # Fill in 

def loglikelihood(theta, y, x): 
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity (in case the optimizer decides to try negatives)
    
    phi= None # fill in

    Phi = None # fill in
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)


    ll =  None # fill in, HINT: you can get indicator functions by using (y>0) and (y==0)

    return ll



def starting_values(y,x): 
    '''starting_values
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N,K = None # fill in
    b_ols = None # fill in
    res = None # fill in
    sig2hat = None # fill in
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Returns E, E_pos
        E: E(y|x)
        E_pos: E(y|x, y>0) 
    '''
    # Fill in 
    E = None
    Epos = None
    return E, Epos

def sim_data(theta, N:int): 
    b = theta[:-1]
    sig = theta[-1]
    K=b.size

    # FILL IN : x will need to contain 1s (a constant term) and randomly generated variables
    xx = None # fill in
    oo = None # fill in
    x  = None # fill in

    eps = None # fill in
    y_lat= None # fill in
    assert y_lat.ndim==1
    y = None # fill in

    return y,x
