import numpy as np 
from scipy.stats import norm
from numpy import linalg as la

name = 'Tobit'

def q(theta, y, x): 
    """
    negative log-likelihood function for the Tobit model
    Args:
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
        y: N x 1 array of dependent variable
        x: N x K array of covariates
    """
    return -loglikelihood(theta, y, x) # Fill in 

def loglikelihood(theta, y, x): 
    """
    log-likelihood function for the Tobit model
    Args:
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
        y: N x 1 array of dependent variable
        x: N x K array of covariates
        returns:
        ll: N x 1 array of log-likelihood contributions
    """
    assert y.ndim == 1, f'y should be 1-dimensional'
    assert theta.ndim == 1, f'theta should be 1-dimensional'

    # unpack parameters 
    b = theta[:-1] # first K parameters are betas, the last is sigma 
    sig = np.abs(theta[-1]) # take abs() to ensure positivity (in case the optimizer decides to try negatives)
    xb_s = x@b / sig
    Phi = norm.cdf(xb_s)
    u_s = (y - x@b)/sig
    phi = norm.pdf(u_s) / sig 
    # avoid taking log of zero
    Phi = np.clip(Phi, 1e-8, 1.-1e-8)
    # log-likelihood function
    ll = (y == 0.0) * np.log(1.0-Phi) + (y > 0) * np.log(phi)
    return ll


def starting_values(y,x): 
    '''starting_values
    Args:
        y: N x 1 array of dependent variable
        x: N x K array of covariates
    Returns
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
    '''
    N,K = x.shape 
    b_ols = la.inv(x.T @ x) @ x.T@y 
    res = y- x@b_ols 
    sig2hat = (1/N-K)*np.dot(res,res) # fill in
    sighat = np.sqrt(sig2hat) # our convention is that we estimate sigma, not sigma squared
    theta0 = np.append(b_ols, sighat)
    return theta0 

def predict(theta, x): 
    '''predict(): the expected value of y given x 
    Args:
        theta: K+1 array, where theta[:K] are betas, and theta[-1] is sigma (not squared)
        x: N x K array of covariates
    Returns E, E_pos
        E: E(y|x)
        E_pos: E(y|x, y>0) 
    '''
    # a. define inputs
    linear_scaled_pred = (x @ theta[:-1]) / theta[-1]
    linear_pred = x @ theta[:-1]
    mills_ratio = norm.pdf(linear_scaled_pred) / norm.cdf(linear_scaled_pred)

    # b. unconditional expectation
    E = norm.cdf(linear_scaled_pred)*linear_pred + theta[-1]*norm.pdf(linear_scaled_pred)
    # c. conditional expectation
    Epos = linear_pred + theta[-1]*mills_ratio
    return E, Epos

def sim_data(theta, N:int): 
    b = theta[:-1]
    sig = theta[-1]
    K=b.size

    # FILL IN : x will need to contain 1s (a constant term) and randomly generated variables
    xx = np.random.normal(size=(N,K-1)) 
    oo = np.ones((N,1)) # intercept
    x  = np.hstack([oo,xx])  # dim N,K

    eps = np.random.normal(scale=sig, size=N)  
    y_lat= x @ b + eps 
    assert y_lat.ndim==1 
    y = np.fmax(0, y_lat) 

    return y,x

def clad(y,x, theta):
    """
    y: N x1 array of dep var
    x: NxK array of covariates
    
    """
    # find residuals
    b = theta[:-1]
    residuals = y- x @b
    # find sum of absolute devs
    mean_deviations = (1/len(y))*np.sum(np.abs(residuals))


